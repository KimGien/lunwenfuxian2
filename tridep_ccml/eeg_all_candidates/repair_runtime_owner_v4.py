"""Native/WSL child admission for new Q9 repair execution; stdlib only.

Actual parent and child identities are captured afresh. No process is stopped,
resumed, or killed. Default import performs no inventory/GPU/record operation.
"""
from __future__ import annotations
import os
from pathlib import Path
import sys
import time
from . import storage as S
from . import qualify_deterministic_repairs_v6 as Q
import json
from . import linux_worker_identity_v2 as L
SOURCE = Path(__file__).resolve()
ROOT = S.OUT / 'continuation_queue_v9'
WORKER_MODULE = 'tridep_ccml.eeg_all_candidates.repair_owned_worker_v4'
QUEUE_MODULE = 'tridep_ccml.eeg_all_candidates.continuation_queue_v9'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def queue_roles(items, pid):
    own = next((x for x in items if x['pid'] == pid), None)
    require(own is not None and QUEUE_MODULE in own['command'], 'Exact native Q9 dispatcher absent')
    roles = [own]
    parent = next((x for x in items if x['pid'] == own['parent']), None)
    if parent:
        require(parent['command'] == own['command'], 'Unknown experimental queue parent')
        roles.append(parent)
    return roles

def validate_native(items, owner, admission, *, worker_pid, platform, linux_snapshot=None, admission_binding=None):
    require(owner.get('id') == admission.get('queue_owner_id') and owner.get('execution_version') == 9 and (owner.get('child_token') == admission.get('token')) and (owner.get('child_job') == admission.get('job')), 'Current actual child/queue lease differs')
    roles = list(owner['native_roles'])
    for role in roles:
        require(any((x == role for x in items)), 'Parent native birth/command/parent changed')
    launch = next((x for x in items if x['pid'] == owner['child_pid']), None)
    require(launch is not None and launch == owner['child_native_identity'] and (launch['parent'] == owner['pid']) and (WORKER_MODULE in launch['command']) and (admission['token'] in launch['command']), 'Exact bound child launcher absent')
    roles.append(launch)
    if platform == 'win32':
        actual = next((x for x in items if x['pid'] == worker_pid), None)
        require(actual is not None, 'Actual Python child identity absent')
        if worker_pid != launch['pid']:
            require(actual['parent'] == launch['pid'] and actual['command'] == launch['command'], 'Native Python child outside exact launcher tree')
            roles.append(actual)
    else:
        require(platform == 'linux' and 'wsl.exe' in launch['command'].lower(), 'Actual WSL launcher required')
        require(admission_binding is not None and owner.get('linux_worker_identity') is not None, 'Controller-captured Linux identity required')
        L.validate_bound(linux_snapshot if linux_snapshot is not None else L.snapshot(), owner['linux_worker_identity'], module=WORKER_MODULE, flags={'--token': admission['token'], '--condition': admission['condition'], '--worker-admission-sha256': admission_binding['sha256'], '--repair-registration-sha256': admission['repair_registration']['sha256']}, worker_pid=worker_pid)
    allowed = {(x['pid'], x['birth']) for x in roles}
    require(all(((x['pid'], x['birth']) in allowed for x in items)), 'Another experimental process remains')
    return roles

def enter(binding, token, *, condition, repair_registration):
    if sys.platform == 'linux':
        require(os.getuid() == 1000, 'Model worker must remain original unprivileged uid1000; root is read-only observer only')
    path = S.checked(binding)
    require(path.is_relative_to((ROOT / 'attempts').resolve()) and path.name == 'worker_admission.json', 'Canonical actual worker admission required')
    admission = S.read(path)
    require(admission['token'] == token and admission['condition'] == condition and (admission['repair_registration'] == repair_registration) and (admission['source'] == S.desc(SOURCE)) and (admission['worker_source'] == S.desc(SOURCE.with_name('repair_owned_worker_v4.py'))), 'Actual worker/source/condition registration differs')
    require(admission['queue_source'] == S.desc(SOURCE.with_name('continuation_queue_v9.py')) and admission['queue_registration'] == S.desc(ROOT / 'registration.json'), 'Actual source7 registration changed')
    registration = S.read(S.checked(admission['queue_registration']))
    require(registration['schema'] == 'eeg_deterministic_repair_queue_v9' and registration['version'] == 9 and (registration['repair_registrations'][condition] == repair_registration), 'Actual Q9 condition job differs')
    for descriptor in registration['sources']:
        S.checked(descriptor)
    deadline = time.monotonic() + 20
    while True:
        Q.check_pause()
        require(not (S.OUT / 'continuation_queue_v5/owner.json').exists(), 'Old Q5 owner remains')
        owner = S.read(ROOT / 'owner.json')
        if owner.get('child_native_identity') and owner.get('child_token') == token and (sys.platform != 'linux' or owner.get('linux_worker_identity')):
            break
        require(time.monotonic() < deadline, 'Parent has not bound actual child birth')
        time.sleep(0.1)
    items = Q.host_inventory()
    roles = validate_native(items, owner, admission, worker_pid=os.getpid(), platform=sys.platform, admission_binding=binding)
    gpu = Q.gpu_empty()
    identity = {'pid': os.getpid(), 'parent': os.getppid(), 'platform': sys.platform, 'native_roles': roles, 'source': admission['worker_source'], 'owner_verifier_source': S.desc(SOURCE), 'queue_owner_id': owner['id'], 'admission': binding, 'GPU_before_runtime': gpu, 'linux_start_ticks': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19] if sys.platform == 'linux' else None, 'linux_worker_identity': owner.get('linux_worker_identity')}
    claim_worker_owner(path.parent / 'worker_owner.json', identity)
    return (admission, identity)

def require_live(admission, identity):
    Q.check_pause()
    require(not (S.OUT / 'continuation_queue_v5/owner.json').exists(), 'Old owner appeared')
    require(S.desc(SOURCE) == identity['owner_verifier_source'], 'Runtime owner verifier changed')
    owner = S.read(ROOT / 'owner.json')
    roles = validate_native(Q.host_inventory(), owner, admission, worker_pid=os.getpid(), platform=sys.platform, admission_binding=identity['admission'])
    require(roles == identity['native_roles'], 'Actual child tree/creation identity changed')
    if sys.platform == 'linux':
        require(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19] == identity['linux_start_ticks'], 'Actual Linux worker birth changed')
    return S.desc(ROOT / 'owner.json')

def claim_worker_owner(path, identity):
    require(path.resolve().is_relative_to((ROOT / 'attempts').resolve()) and path.name == 'worker_owner.json', 'Canonical worker claim path required')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(identity, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
