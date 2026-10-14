"""Hold the current recovered Q5 dispatcher at a natural full-CSBrain boundary.

Only explicit --inspect reads process identities. --register writes a separate
control manifest. Only --arm may suspend/resume the exact inspected dispatcher.
Existing training children are never suspended or terminated; the same parent
is resumed once after optional zero-update probes. No new queue is launched.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tridep_ccml.eeg_all_candidates import storage as S, registry as R, continuation_queue as Q1
from tridep_ccml.eeg_all_candidates import continuation_queue_v5 as Q5, lost_f3_closure_v1 as L
import sync_qualification_boundary_guard_v1 as G1
import sync_qualification_boundary_guard_v2 as G2

SOURCE = Path(__file__).resolve()
ROOT = S.OUT / 'execution_acceleration_v1/sync_boundary_v3'
REGISTRATION = ROOT / 'registration.json'
BIRTH_OBSERVATION = Q5.ROOT / 'recovery_launches/native_birth_observation_1791597115326.json'
EXTERNAL_GRAPHICS = S.OUT / 'execution_acceleration_v1/sync_v3_external_qq_observation_1791598770508.json'
CONDITION = 'CSBrain-lastblock-finetune'
ROLES = ('dispatcher', 'launcher', 'gpu_child')
QUALIFIERS = ('qualify_sync_reduction_v3.py', 'qualify_training_sync_v3.py')
DELEGATE_PINS = {Path(G1.__file__): '3a40f7b1d13b0bb85e868376f72c809e3d938d676ad9270be894e28a768308d3',
                 Path(G2.__file__): 'becadd7aadb20a9ca415d661f9aa2291e7c62de4762d4159dd6dbeaec8e6199c',
                 Path(L.__file__): '97dd30a04a0f028890c124001a4f06e4f2b22cf86677a3d6f0731fd53b790a57'}
for _path, _sha in DELEGATE_PINS.items():
    if hashlib.sha256(_path.read_bytes()).hexdigest() != _sha:
        raise ValueError('Reviewed control delegate changed: ' + str(_path))


def require(value, message):
    if not value: raise ValueError(message)


UserPauseRequested = G1.UserPauseRequested


def require_no_user_pause():
    for name in ('sync_boundary', 'sync_boundary_v2', 'sync_boundary_v3'):
        marker = S.OUT / 'execution_acceleration_v1' / name / 'user_pause.json'
        if marker.exists():
            value = S.read(marker)
            require(type(value) is dict and set(value) == {'paused'} and value['paused'] is True,
                    'Invalid durable user-pause control; no automatic action allowed')
            raise UserPauseRequested('User revoked new probes and automatic dispatcher resume')


def write_exclusive(path, value):
    """Real create-exclusive lease, never exists-then-replace."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def snapshot_exact(binding, destination):
    original = S.checked(binding); data = Path(original).read_bytes()
    require(hashlib.sha256(data).hexdigest() == binding['sha256'], 'Original bytes changed while archiving control evidence')
    with Path(destination).open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    saved = S.desc(destination)
    require(saved['sha256'] == binding['sha256'] and S.desc(original) == binding, 'Exact control snapshot raced')
    return saved


def recovery_lineage():
    lineage = L.verify_recovery_registration()
    require(len(lineage['value']['sources']) == 80, 'Exact original recovered80 sources required')
    owner = S.read(Q5.ROOT / 'owner.json')
    require(owner.get('source') == S.desc(Q5.SOURCE)
            and owner.get('execution_registration') == lineage['base_queue_registration']
            and type(owner.get('execution_version')) is int and owner['execution_version'] == 5
            and owner.get('recovery_registration') == lineage['registration']
            and owner.get('recovery_execution_source') == lineage['source'], 'Actual recovered Q5 owner provenance differs')
    closure = L.read_json(owner['recovery_closure'])
    proof = L.verify_lost_f3_attempt(closure['original_attempt'], L.read_json(closure['original_attempt']),
                                   {key: closure[key] for key in ('condition', 'seed', 'fold')})
    require(proof is not None and proof['passed'] is True and proof['closure'] == owner['recovery_closure']
            and proof['recovery_registration'] == lineage['registration'], 'Actual recovered Q5 closure differs')
    return lineage, owner


def boundary(roles=None, *, full=True):
    lineage, owner = recovery_lineage() if full else (None, S.read(Q5.ROOT / 'owner.json'))
    require(owner.get('child_job') == CONDITION + '-fit', 'Current queue is no longer the registered CSBrain job')
    if roles is not None:
        require(owner['pid'] == roles['dispatcher']['pid'] and owner['child_pid'] == roles['launcher']['pid'],
                'Actual current Q5 process boundary changed')
    path = (S.WORK / owner['linux_owner']).resolve().with_name('status.json')
    require(path.is_relative_to((Q5.ROOT / 'attempts').resolve()), 'Current Q5 attempt escaped its directory')
    attempt = S.read(path)
    require(attempt.get('status') == 'running' and attempt.get('child_pid') == owner['child_pid']
            and attempt.get('job', {}).get('module') == 'finetune_v3'
            and attempt['job']['id'] == CONDITION + '-fit', 'Current Q5 child route changed')
    return owner, path, lineage


class DynamicNative(G1.H2.WindowsBackend):
    """Read-only child handles; dispatcher alone receives suspend/resume rights."""
    def __init__(self, roles):
        super().__init__(); self.roles = roles

    def capture(self, role, *, arm=False):
        expected = self.roles[role]
        require(not arm or role == 'dispatcher', 'Training child control is forbidden')
        rights = 0x1000 | 0x100000 | (0x0800 if arm else 0)
        handle = self.kernel.OpenProcess(rights, False, expected['pid'])
        require(handle, 'Cannot capture exact ' + role)
        try:
            times = [ctypes.c_ulonglong() for _ in range(4)]
            require(self.kernel.GetProcessTimes(handle, *(ctypes.byref(x) for x in times)), 'Native birth unavailable')
            created = (datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=times[0].value // 10)).astimezone(timezone(timedelta(hours=8))).isoformat()
            size = ctypes.c_ulong(32768); image = ctypes.create_unicode_buffer(size.value)
            require(self.kernel.QueryFullProcessImageNameW(handle, 0, image, ctypes.byref(size)), 'Image identity unavailable')
            value = {'role': role, 'pid': expected['pid'], 'parent_pid': self.parents().get(expected['pid']),
                     'creation_100ns': str(times[0].value), 'created_local': created, 'image': image.value}
            require(value['creation_100ns'] == expected['creation_100ns']
                    and value['parent_pid'] == expected['parent_pid'], 'Exact native birth or parent changed: ' + role)
            if 'image' in expected: require(value == expected, 'Inspected process identity changed: ' + role)
            return {**value, 'handle': handle}
        except BaseException:
            self.kernel.CloseHandle(handle); raise

    def descendants(self):
        return G1.qualifier_descendants(self, self.roles['dispatcher']['pid'])

    def suspend_dispatcher(self, ref):
        require(ref['role'] == 'dispatcher' and G1.H2.public_identity(ref) == self.roles['dispatcher'], 'Only inspected dispatcher may be held')
        require(self.ntdll.NtSuspendProcess(ref['handle']) == 0, 'Native dispatcher suspension failed')

    def terminate_dispatcher(self, ref):
        raise ValueError('This guardian never terminates a formal process')


def backend(roles):
    return DynamicNative(roles)


def experimental_sources():
    paths = [SOURCE, Path(G1.__file__), Path(G2.__file__), Path(G1.H2.__file__)]
    paths += [S.REPORT / name for name in QUALIFIERS]
    paths += [S.REPORT / name for name in ('qualify_sync_reduction_v2.py', 'qualify_training_sync_v2.py',
              'qualify_sync_reduction_v1.py', 'qualify_training_sync_v1.py', 'qualify_acceleration_v2.py')]
    paths += [S.WORK / 'tridep_ccml/eeg_all_candidates' / name for name in
              ('sync_qualification_gate_v3.py', 'sync_qualification_gate_v2.py', 'sync_qualification_gate_v1.py',
               'sync_reduction_v1.py', 'sync_state_digest_v1.py', 'sync_subject_update_v1.py')]
    return list({str(path.resolve()): S.desc(path) for path in paths}.values())


def inspect():
    require_no_user_pause()
    owner, attempt_path, lineage = boundary()
    births = S.read(BIRTH_OBSERVATION)
    require(births.get('schema') == 'recovered_q5_native_birth_observation_v1'
            and births.get('owner_id') == owner['id'], 'Actual native birth observation belongs to another owner')
    by_pid = {row['pid']: row for row in births['processes']}
    require(len(by_pid) == 4 and all(row.get('has_exited') is False for row in by_pid.values()), 'Expected original four native birth observations')
    observer = G1.H2.WindowsBackend(); parents = observer.parents()
    dispatcher, launcher = owner['pid'], owner['child_pid']
    children = [pid for pid, parent in parents.items() if parent == launcher]
    require(len(children) == 1, 'Launcher must have exactly one actual Python child')
    pids = dict(dispatcher=dispatcher, launcher=launcher, gpu_child=children[0])
    expected_parents = dict(dispatcher=parents.get(dispatcher), launcher=dispatcher, gpu_child=launcher)
    require(set(pids.values()) | {expected_parents['dispatcher']} == set(by_pid), 'Native birth chain differs')
    roles = {role: {'role': role, 'pid': pid, 'parent_pid': expected_parents[role],
                    'creation_100ns': by_pid[pid]['creation_100ns']} for role, pid in pids.items()}
    native = backend(roles); refs = []
    try:
        for role in ROLES: refs.append(native.capture(role))
        require(all(not native.exited(ref) for ref in refs), 'Current training boundary has already moved')
        require(native.descendants() == {launcher, pids['gpu_child']}, 'Unexpected current training child')
        require(boundary(roles)[0] == owner, 'Owner changed during inspection')
        value = {'schema': 'sync_q5_dynamic_boundary_inspection_v3', 'source': S.desc(SOURCE),
                 'queue_owner': S.desc(Q5.ROOT / 'owner.json'), 'queue_attempt': S.desc(attempt_path),
                 'registration': lineage['base_queue_registration'], 'recovery_registration': lineage['registration'],
                 'recovery_closure': owner['recovery_closure'], 'native_birth_observation': S.desc(BIRTH_OBSERVATION),
                 'roles': {ref['role']: G1.H2.public_identity(ref) for ref in refs},
                 'checked_at': S.now(), 'process_actions': 0}
        path = ROOT / ('inspection_' + str(time.time_ns()) + '.json'); write_exclusive(path, value)
        return S.desc(path)
    finally:
        for ref in refs: native.close(ref)


def register_control(expected_binding):
    require_no_user_pause()
    expected = S.read(S.checked(expected_binding))
    require(expected.get('schema') == 'sync_q5_dynamic_boundary_inspection_v3' and expected['source'] == S.desc(SOURCE), 'Reviewed V3 inspection required')
    owner, _, lineage = boundary(expected['roles'])
    require(expected['queue_owner'] == S.desc(Q5.ROOT / 'owner.json')
            and expected['registration'] == lineage['base_queue_registration']
            and expected['recovery_registration'] == lineage['registration']
            and expected['recovery_closure'] == owner['recovery_closure'], 'Inspection lineage changed')
    value = {'schema': 'recovered_q5_sync_control_registration_v3', 'version': 3, 'source': S.desc(SOURCE),
             'expected': expected_binding, 'registration': expected['registration'],
             'recovery_registration': lineage['registration'], 'recovery_closure': owner['recovery_closure'],
             'queue_owner': expected['queue_owner'], 'roles': expected['roles'],
             'queue_attempt': expected['queue_attempt'],
             'native_birth_observation': expected['native_birth_observation'],
             'external_graphics_evidence': S.desc(EXTERNAL_GRAPHICS),
             'protected_sources': lineage['value']['sources'], 'experimental_sources': experimental_sources(),
             'same_dispatcher_resume_only': True, 'formal_training_math_changed': False,
             'formal_training_process_termination_permitted': False, 'registered_at': S.now()}
    for item in value['protected_sources'] + value['experimental_sources']: S.checked(item)
    external_graphics_evidence()
    value['queue_owner_snapshot'] = snapshot_exact(value['queue_owner'], ROOT / 'queue_owner_snapshot.json')
    value['queue_attempt_snapshot'] = snapshot_exact(value['queue_attempt'], ROOT / 'queue_attempt_snapshot.json')
    write_exclusive(REGISTRATION, value)
    return S.desc(REGISTRATION)


def verify_control(binding=None):
    binding = S.desc(REGISTRATION) if binding is None else binding
    require(Path(S.checked(binding)).resolve() == REGISTRATION.resolve(), 'Control registration path differs')
    value = S.read(REGISTRATION)
    require(value.get('schema') == 'recovered_q5_sync_control_registration_v3' and type(value.get('version')) is int
            and value['version'] == 3 and value['source'] == S.desc(SOURCE), 'Control source/version differs')
    expected = S.read(S.checked(value['expected']))
    owner, _, lineage = boundary(value['roles'])
    require(value['registration'] == lineage['base_queue_registration']
            and value['recovery_registration'] == lineage['registration'] and value['recovery_closure'] == owner['recovery_closure']
            and value['protected_sources'] == lineage['value']['sources'] and value['experimental_sources'] == experimental_sources()
            and value['queue_owner'] == S.desc(Q5.ROOT / 'owner.json')
            and value['roles'] == expected['roles'] and expected['source'] == value['source']
            and value['queue_attempt'] == expected['queue_attempt']
            and value['queue_owner_snapshot']['sha256'] == value['queue_owner']['sha256']
            and value['queue_attempt_snapshot']['sha256'] == value['queue_attempt']['sha256']
            and value.get('same_dispatcher_resume_only') is True and value.get('formal_training_math_changed') is False
            and value.get('formal_training_process_termination_permitted') is False
            and value['external_graphics_evidence'] == S.desc(EXTERNAL_GRAPHICS),
            'Control registration or exact owner changed')
    for item in value['protected_sources'] + value['experimental_sources'] + [value['native_birth_observation'], value['external_graphics_evidence'],
            value['queue_owner_snapshot'], value['queue_attempt_snapshot'], value['queue_attempt']]: S.checked(item)
    return value


def completed_artifacts():
    return G2.completed_artifacts()


def gpu_observation():
    return G2.gpu_observation()


def external_graphics_evidence():
    value = S.read(EXTERNAL_GRAPHICS)
    process = value['process']
    require(value.get('schema') == 'sync_v3_observed_external_qq_gpu_process_v1'
            and value.get('nvidia_observed_type') == 'C+G'
            and value.get('native_creation_100ns') == '134360680926134212'
            and type(process.get('ProcessId')) is int and process['ProcessId'] == 15304
            and type(process.get('ParentProcessId')) is int and process['ParentProcessId'] == 34776
            and process.get('ExecutablePath', '').replace('\\', '/').casefold() == 'c:/program files/tencent/qqnt/qq.exe'
            and value.get('nvidia_observed_path', '').casefold() == 'c:/program files/tencent/qqnt/qq.exe'
            and '--type=gpu-process' in process.get('CommandLine', '').split()
            and type(value.get('model_or_process_control_operations')) is int and value['model_or_process_control_operations'] == 0,
            'Exact read-only QQ graphics evidence required')
    return value


def external_native_identity(expected):
    native = backend({'external_graphics': expected}); ref = None
    try:
        ref = native.capture('external_graphics')
        require(not native.exited(ref), 'Observed QQ process has exited; new observation required')
        return G1.H2.public_identity(ref)
    finally:
        if ref is not None: native.close(ref)


def verify_external_graphics_context(context):
    evidence = external_graphics_evidence(); original = evidence['process']
    require(str(context.get('pid')) == str(original['ProcessId']) and context.get('type') == 'C+G'
            and str(context.get('process_name', '')).replace('\\', '/').casefold()
                == original['ExecutablePath'].replace('\\', '/').casefold(), 'Unverified external C+G object')
    expected = {'role': 'external_graphics', 'pid': original['ProcessId'], 'parent_pid': original['ParentProcessId'],
                'creation_100ns': evidence['native_creation_100ns']}
    actual = external_native_identity(expected)
    require(all(actual.get(key) == value for key, value in expected.items())
            and actual.get('image', '').replace('\\', '/').casefold()
                == original['ExecutablePath'].replace('\\', '/').casefold(), 'External QQ PID/birth/path changed')
    return {'evidence': S.desc(EXTERNAL_GRAPHICS), 'actual_native_identity': actual,
            'classification': 'previously observed exact QQ graphics process, never a performance-idle GPU'}


def training_gpu_scope(observation):
    require(type(observation.get('contexts')) is list, 'GPU context inventory unavailable')
    graphics = []
    for context in observation['contexts']:
        if context.get('type') == 'G': continue
        graphics.append(verify_external_graphics_context(context))
    return {'passed': True, 'contexts': observation['contexts'], 'observation': observation,
            'verified_external_graphics': graphics,
            'scope': 'single EEG compute worker; pure graphics and exact source-bound QQ object may coexist',
            'optional_performance_measurement_quiet': not observation['contexts']}


def guardian_identity(native):
    return {**G1.self_identity(native), 'source': S.desc(SOURCE)}


def resume_dispatcher(native, ref):
    require_no_user_pause()
    require(ref['role'] == 'dispatcher' and not native.exited(ref), 'Exact original dispatcher unavailable')
    native.ntdll.NtResumeProcess.argtypes = (ctypes.c_void_p,)
    native.ntdll.NtResumeProcess.restype = ctypes.c_long
    require(native.ntdll.NtResumeProcess(ref['handle']) == 0, 'Native dispatcher resume failed')


def require_probe_cleanup(native, process):
    if process is not None:
        require(process.poll() is not None, 'Optional qualifier remains alive, even if GPU is idle')
        require(not G1.qualifier_descendants(native, process.pid), 'Optional CPU/CUDA descendant remains')
    require(not (S.OUT / 'execution_acceleration_v1/sync_qualification_owner.json').exists(), 'Optional qualification lease remains')
    return training_gpu_scope(gpu_observation())


def observe_probe_gpu(native, process, monitor):
    """Sample exclusivity; unobserved intervals never become a clean certificate."""
    started = time.monotonic()
    try:
        allowed = {process.pid} | G1.qualifier_descendants(native, process.pid)
        observed = gpu_observation()
        require(type(observed.get('contexts')) is list, 'GPU monitor inventory unavailable')
        external = [item for item in observed['contexts'] if str(item.get('pid')) not in {str(pid) for pid in allowed}]
        monitor['samples'].append({'started_monotonic': started, 'returned_monotonic': time.monotonic(),
                                   'allowed_current_qualifier_pids': sorted(allowed), 'observation': observed,
                                   'external_contexts': external})
        if external: monitor['external_context_observed'] = True
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)): raise
        monitor['observation_errors'].append({'started_monotonic': started, 'error_type': type(error).__name__, 'error': str(error)})
    finally:
        monitor['observer_query_seconds_observed'] += time.monotonic() - started


def wait_probe(process, *, native=None, monitor=None, on_observation=None, timeout=1200):
    deadline = time.monotonic() + timeout
    next_observation = time.monotonic()
    while True:
        require_no_user_pause()
        if native is not None and time.monotonic() >= next_observation:
            observe_probe_gpu(native, process, monitor)
            next_observation = time.monotonic() + 2.0
            if on_observation is not None: on_observation()
            require_no_user_pause()
        code = process.poll()
        if code is not None: return code
        require(time.monotonic() < deadline, 'Optional qualifier timeout; dispatcher stays held while it is live')
        time.sleep(0.25)


def verify_live_boundary(binding):
    require_no_user_pause()
    proof = S.read(S.checked(binding))
    require(proof.get('schema') == 'sync_qualification_safe_boundary_v3' and proof.get('passed') is True
            and proof.get('source') == S.desc(SOURCE) and proof.get('queue') == 'continuation_queue_v5', 'Reviewed V3 live boundary required')
    control = verify_control(proof['control_registration'])
    require(proof['registration'] == control['registration'] and proof['recovery_registration'] == control['recovery_registration']
            and proof['recovery_closure'] == control['recovery_closure']
            and proof['queue_owner'] == control['queue_owner'] and proof['experimental_sources'] == control['experimental_sources']
            and proof['held_process_identities'] == control['roles'], 'Proof differs from dynamic control registration')
    require(proof['guardian_owner'] == S.desc(ROOT / 'guardian_owner.json')
            and proof['guardian_state_path'] == (ROOT / 'state.json').relative_to(S.WORK).as_posix(), 'Guardian paths differ')
    lease = S.read(S.checked(proof['guardian_owner'])); state = S.read(ROOT / 'state.json')
    require(state.get('phase') == 'qualification_running' and state.get('dispatcher_suspended') is True
            and state.get('boundary_proof') == binding and state.get('source') == proof['source']
            and state.get('guardian_identity') == proof['guardian_identity']
            and state.get('held_process_identities') == proof['held_process_identities']
            and lease.get('identity') == proof['guardian_identity'] and lease.get('control_registration') == proof['control_registration']
            and proof['guardian_identity'].get('source') == proof['source'], 'Published suspension lease/state is not live')
    require(proof.get('gpu_children_exit_verified') is True and type(proof.get('completed_CSBrain_fits_verified')) is int
            and proof['completed_CSBrain_fits_verified'] == 15
            and all(type(proof.get(key)) is int and proof[key] == 0 for key in ('launcher_exit_code', 'gpu_child_exit_code')),
            'Actual full15 and both normal child exits required')
    native = backend(control['roles']); ref = None
    try:
        G1.current_process_identity(native, proof['guardian_identity'])
        ref = native.capture('dispatcher')
        require(not native.exited(ref) and G1.H2.public_identity(ref) == control['roles']['dispatcher']
                and not native.descendants(), 'Held parent changed or training descendant remains')
        require(completed_artifacts() == proof['completed_CSBrain_artifacts'], 'Full15 artifacts changed')
        require_no_user_pause()
        return {'passed': True, 'process_control_calls': 0, 'gpu_children_exit_verified': True,
                'held_dispatcher_running_suspended': True,
                'suspension_basis': 'source-bound live guardian recorded successful native suspension and retains exact dispatcher handle'}
    finally:
        if ref is not None: native.close(ref)


def arm(expected_binding):
    require_no_user_pause()
    control_binding = S.desc(REGISTRATION); control = verify_control(control_binding)
    require(control['expected'] == expected_binding, 'Arm inspection differs from registered immutable identity')
    native = backend(control['roles']); refs = []; held = False; resumed = False; suspend_attempted = False; resume_attempted = False; process = None
    identity = guardian_identity(native); state_path = ROOT / 'state.json'; lease_path = ROOT / 'guardian_owner.json'
    write_exclusive(lease_path, {'identity': identity, 'source': S.desc(SOURCE), 'expected': expected_binding,
                                'control_registration': control_binding, 'started_at': S.now()})
    state = {'source': S.desc(SOURCE), 'phase': 'capturing_exact_handles', 'passed': False, 'dispatcher_suspended': False,
             'guardian_identity': identity, 'control_registration': control_binding, 'held_process_identities': control['roles'],
             'GPU_training_child_suspend_or_terminate_calls': 0, 'formal_process_terminate_calls': 0,
             'formal_queue_spawn_calls': 0, 'formal_training_math_changed': False, 'actual_sync_optimization_applied': False,
             'dispatcher_resume_calls_returned': 0, 'outcomes': [], 'experimental_sources': control['experimental_sources'],
             'control_timing_scope': 'Observed guardian wall intervals overlap queue process.wait and nested probes; never formal GPU busy time or additive training cost',
             'hold_started_monotonic': None, 'hold_ended_monotonic': None, 'hold_seconds': None,
             'children_wait_ended_monotonic': None, 'children_wait_seconds': None, 'probe_intervals': []}
    try:
        owner, _, _ = boundary(control['roles'])
        for role in ROLES:
            ref = native.capture(role, arm=role == 'dispatcher'); refs.append(ref)
            require(G1.H2.public_identity(ref) == control['roles'][role], 'Exact role changed before suspension')
        require(all(not native.exited(ref) for ref in refs), 'Child exit raced with capture; do not suspend a changed job')
        require(native.descendants() == {control['roles'][key]['pid'] for key in ('launcher', 'gpu_child')}, 'Unexpected training descendant before hold')
        require_no_user_pause(); suspend_attempted = True
        native.suspend_dispatcher(refs[0]); held = True
        state['hold_started_monotonic'] = time.monotonic()
        require(boundary(control['roles'])[0] == owner, 'Owner changed across successful suspension')
        require(native.descendants().issubset({control['roles'][key]['pid'] for key in ('launcher', 'gpu_child')}), 'Job changed across suspension')
        state.update(phase='waiting_for_CSBrain_full15', dispatcher_suspended=True, current_GPU_training_continues=True,
                     suspension_basis='successful NtSuspendProcess on exact inspected dispatcher', waiting_started_at=S.now())
        S.save(state_path, state)
        while not (native.exited(refs[1]) and native.exited(refs[2])):
            require_no_user_pause()
            require(boundary(control['roles'], full=False)[0] == owner, 'Held owner changed while waiting')
            require(native.descendants().issubset({control['roles'][key]['pid'] for key in ('launcher', 'gpu_child')}), 'Unexpected child while draining')
            time.sleep(2)
        require_no_user_pause()
        require(native.exit_code(refs[1]) == native.exit_code(refs[2]) == 0, 'Original full CSBrain job did not exit normally')
        state['children_wait_ended_monotonic'] = time.monotonic()
        state['children_wait_seconds'] = state['children_wait_ended_monotonic'] - state['hold_started_monotonic']
        artifacts = completed_artifacts(); require(len(artifacts) == 90, 'All90 complete15 artifact bindings required')
        verify_control(control_binding)
        observation = require_probe_cleanup(native, process)['observation']
        if observation['contexts']:
            deferred = ROOT / ('optional_sync_qualification_deferred_' + str(time.time_ns()) + '.json')
            write_exclusive(deferred, {'source': S.desc(SOURCE), 'control_registration': control_binding,
                                      'status': 'deferred_external_GPU_contexts', 'observed_GPU_contexts': observation,
                                      'model_constructions': 0, 'GPU_forward_calls': 0, 'backward_calls': 0,
                                      'optimizer_updates': 0, 'formal_fits': 0, 'GPU_speedup_claimed': False})
            state['deferred_qualification'] = S.desc(deferred)
        else:
            folder = R.fit_directory(CONDITION, 22, 5)
            proof = {'schema': 'sync_qualification_safe_boundary_v3', 'passed': True, 'source': S.desc(SOURCE),
                     'queue': 'continuation_queue_v5', 'registration': control['registration'],
                     'original_training_registration': R.registration_descriptor(), 'recovery_registration': control['recovery_registration'],
                     'recovery_closure': control['recovery_closure'], 'control_registration': control_binding,
                     'queue_owner': control['queue_owner'], 'held_process_identities': control['roles'],
                     'gpu_children_exit_verified': True, 'launcher_exit_code': 0, 'gpu_child_exit_code': 0,
                     'held_dispatcher_running_suspended': True, 'gpu_idle': True, 'zero_optimizer_probe_only': True,
                     'completed_CSBrain_fits_verified': 15, 'completed_CSBrain_artifacts': artifacts,
                     'condition': CONDITION, 'seed': 22, 'fold': 5, 'registered_fold_ids': R.load_registration()['folds']['fold_5'],
                     'checkpoint': S.desc(folder / 'latest.pt'), 'train_history': S.desc(folder / 'train_history.json'),
                     'guardian_identity': identity, 'guardian_owner': S.desc(lease_path),
                     'guardian_state_path': state_path.relative_to(S.WORK).as_posix(),
                     'experimental_sources': control['experimental_sources'], 'created_at': S.now()}
            proof_path = ROOT / ('boundary_' + str(time.time_ns()) + '.json'); write_exclusive(proof_path, proof)
            binding = S.desc(proof_path)
            state.update(phase='qualification_running', current_GPU_training_continues=False, boundary_proof=binding)
            S.save(state_path, state); verify_live_boundary(binding)
            for name in QUALIFIERS:
                require_no_user_pause(); require_probe_cleanup(native, process); verify_control(control_binding)
                if gpu_observation()['contexts']:
                    state['remaining_optional_qualification_deferred_after_GPU_race'] = True; break
                require_no_user_pause()
                log_path = ROOT / (name.removesuffix('.py') + '.log')
                interval = {'qualifier': name, 'started_monotonic': time.monotonic(), 'ended_monotonic': None,
                            'seconds': None, 'scope': 'Launcher lifetime plus wrapper wait; overlaps primary probe ledger'}
                monitor = {'sampling_interval_seconds': 2.0, 'samples': [], 'external_context_observed': False,
                           'observation_errors': [], 'observer_query_seconds_observed': 0.0,
                           'continuous_exclusivity_verified': False, 'between_sample_exclusivity': 'unknown',
                           'observer_scope': 'Read-only GPU inventory/ancestry checks during probe; observer overhead may perturb timings, not formal model work'}
                interval['performance_exclusivity_monitor'] = monitor
                state['probe_intervals'].append(interval)
                S.save(state_path, state)
                with log_path.open('ab') as log:
                    process = subprocess.Popen([str(S.WORK / 'S00/.venv/Scripts/python.exe'), '-B', '-u', str(S.REPORT / name),
                                                '--run', '--boundary-proof', str(proof_path), '--boundary-sha', binding['sha256']],
                                               cwd=S.WORK, stdout=log, stderr=subprocess.STDOUT, creationflags=0x08000000)
                    state.update(active_qualifier=name, active_qualifier_launcher_pid=process.pid); S.save(state_path, state)
                    code = wait_probe(process, native=native, monitor=monitor, on_observation=lambda: S.save(state_path, state))
                interval['ended_monotonic'] = time.monotonic()
                interval['seconds'] = interval['ended_monotonic'] - interval['started_monotonic']
                require_no_user_pause(); cleanup = require_probe_cleanup(native, process)
                monitor['post_probe_GPU_observation'] = cleanup['observation']
                if cleanup['contexts']: monitor['external_context_observed'] = True
                sampled_clean = bool(monitor['samples']) and not monitor['external_context_observed'] and not monitor['observation_errors']
                monitor['sampled_exclusivity_passed'] = sampled_clean
                monitor['performance_contamination_observed_or_unknown'] = not sampled_clean
                state['outcomes'].append({'qualifier': name, 'returncode': code, 'log': S.desc(log_path),
                                          'qualifier_report_passed': code == 0,
                                          'qualification_passed': code == 0 and sampled_clean, 'qualification_declined': code == 2,
                                          'performance_sampled_clean': sampled_clean, 'continuous_performance_clean_verified': False,
                                          'performance_between_samples_unknown': True,
                                          'speed_results_eligible_for_runtime_activation': False,
                                          'execution_failed': code not in (0, 2)})
                process = None; state.update(active_qualifier=None, active_qualifier_launcher_pid=None); S.save(state_path, state)
                if code not in (0, 2): break
        require_no_user_pause(); require_probe_cleanup(native, process); verify_control(control_binding)
        require(completed_artifacts() == artifacts, 'Full15 evidence changed before resume')
        require(not native.descendants(), 'A training descendant appeared before resume')
        require_no_user_pause(); resume_attempted = True
        resume_dispatcher(native, refs[0]); held = False; resumed = True
        state['hold_ended_monotonic'] = time.monotonic()
        state['hold_seconds'] = state['hold_ended_monotonic'] - state['hold_started_monotonic']
        state.update(dispatcher_resume_calls_returned=1, dispatcher_suspended=False,
                     phase='qualification_finished_same_Q5_resumed', passed=True, boundary_workflow_completed=True,
                     all_qualifications_passed=len(state['outcomes']) == len(QUALIFIERS) and all(x['qualification_passed'] for x in state['outcomes']),
                     finished_at=S.now(), same_dispatcher_resumed=True,
                     queue_attempt_elapsed_includes_dispatcher_hold=True,
                     queue_attempt_seconds_not_formal_or_additive_probe_time=True)
        # held/resumed update precedes I/O: a failed write cannot resume twice.
        S.save(state_path, state)
        result = ROOT / ('same_dispatcher_resume_complete_' + str(time.time_ns()) + '.json')
        write_exclusive(result, {**state, 'completed_CSBrain_artifacts': artifacts, 'native_handle_identity': control['roles']['dispatcher']})
        return state
    except BaseException as error:
        state.update(passed=False, error_type=type(error).__name__, error=str(error), dispatcher_suspended=held)
        try:
            require_no_user_pause()
        except BaseException as pause_error:
            state.update(phase='user_pause_Q5_held' if held else 'user_pause_before_suspend',
                         automatic_resume_revoked=True, control_error=str(pause_error))
            if suspend_attempted and not held and not resumed:
                state.update(phase='user_pause_suspend_outcome_unconfirmed', dispatcher_suspended=None)
            S.save(state_path, state); return state
        if resumed:
            state.update(phase='same_Q5_resumed_record_write_failed', same_dispatcher_resumed=True,
                         dispatcher_resume_calls_returned=1, dispatcher_suspended=False)
        elif resume_attempted:
            state.update(phase='same_Q5_resume_outcome_unconfirmed', dispatcher_suspended=None,
                         dispatcher_resume_calls_started=1, dispatcher_resume_calls_returned=None,
                         automatic_resume_retry_forbidden=True)
        elif suspend_attempted and not held:
            state.update(phase='same_Q5_suspend_outcome_unconfirmed', dispatcher_suspended=None,
                         automatic_suspend_or_resume_retry_forbidden=True)
        elif held:
            try:
                require_no_user_pause()
                verify_control(control_binding)
                require(boundary(control['roles'])[0] == owner, 'Exact held owner changed; no recovery resume')
                require(native.descendants().issubset({control['roles'][key]['pid'] for key in ('launcher', 'gpu_child')}),
                        'Unexpected training child; no recovery resume')
                if native.exited(refs[2]): require_probe_cleanup(native, process)
                require_no_user_pause(); resume_attempted = True
                resume_dispatcher(native, refs[0]); held = False; resumed = True
                state['hold_ended_monotonic'] = time.monotonic()
                state['hold_seconds'] = state['hold_ended_monotonic'] - state['hold_started_monotonic']
                state.update(phase='optional_guard_failed_same_Q5_resumed', dispatcher_suspended=False,
                             same_dispatcher_resumed=True, dispatcher_resume_calls_returned=1)
            except BaseException as cleanup_error:
                state.update(phase='same_Q5_resume_outcome_unconfirmed' if resume_attempted else 'cleanup_unverified_Q5_held',
                             cleanup_error=str(cleanup_error), dispatcher_suspended=None if resume_attempted else True,
                             automatic_resume_retry_forbidden=resume_attempted)
        else: state['phase'] = 'refused_before_suspend'
        S.save(state_path, state); raise
    finally:
        for ref in refs: native.close(ref)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--inspect', action='store_true'); group.add_argument('--register', action='store_true'); group.add_argument('--arm', action='store_true')
    parser.add_argument('--expected'); parser.add_argument('--expected-sha256')
    args = parser.parse_args()
    if args.register or args.arm:
        require(args.expected and args.expected_sha256, 'Exact reviewed inspection path and SHA required')
        expected = {'path': args.expected, 'sha256': args.expected_sha256}
        print(json.dumps(register_control(expected) if args.register else arm(expected)))
    else: print(json.dumps(inspect()))
