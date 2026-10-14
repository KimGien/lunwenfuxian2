"""Separate deterministic-repair engineering; default is zero-operation description.

No current queue/source/fit status is changed. --run requires the current Q5
owner to have released, an actual failed/incomplete terminal with preserved
lineage, no other experimental processes, and an empty GPU process inventory.
CodeBrain runs in the Windows environment; EEGMamba runs in the pinned WSL
environment. A permanent condition intent prevents blind repeats after failure.
Successful engineering is not a formal fit or permission to fake Q5 completion.
"""
from __future__ import annotations
import argparse
import base64
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types
import uuid
import xml.etree.ElementTree as ET
from . import linux_worker_identity_v2 as L
from . import native_launcher_tree_v1 as N
WORK = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve()
OUT = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
ROOT = OUT / 'deterministic_repair_qualification_v7'
QUEUE = OUT / 'continuation_queue_v5'
REL = 'tridep_ccml/eeg_all_candidates/'
CONDITIONS = ('CodeBrain-lastblock-finetune', 'EEGMamba-lastblock-finetune')
MANIFEST_SHA = '653df2183b1a9cdcd74755f69629d2cf89ca3de8a6b7e9bf26b7df0bfcabbaf4'
MATRIX_SHA = '27bf2972c08f77457959cf3cba5bc6056c320f223c2426af5da3ae88d55ce090'
PINS = {REL + 'finetune.py': 'cda19474654233c95072acae475174d9011bb0739be650a597194f04c4b0aa24', REL + 'adapters/codebrain_v3.py': '0e2d9ee14c2b0d3855532e4bca152c725703838138a8e6ef13b8b0051cace767', REL + 'codebrain_linear_backward_cpu_v1.py': '7e98dfbfaa8f85ef062f6dcee45e698beaf224ceaab65fda0c8e01da5527881d', REL + 'adapters/eegmamba_v2.py': 'c516c843db45ab987da560d1c952de1d888d0f873e57e5b2c262aa1c0b655741', REL + 'mamba_private_backward_graph_v1.py': '417c2440d0ff5f1be7ea923d2f81bfff56faa3d8444019013b0df246a0efab5a', REL + 'mamba_cumsum_cpu_fp32_v1.py': '0159cec62acd3c7d07d60bbdeef961110e28d5b5579d47267655399da4e3c0e4', 'tridep_ccml/eeg_all_candidates/linux_worker_identity_v2.py': 'abe21d44b8bc663a6506e1d996c67db45212b4a6d5f4c42e4d6a8bc3cd0aec57'}
PINS[REL + 'native_launcher_tree_v1.py'] = '09ad696871414baa47e9505ab450ca5ca7e5d2c62599c34dd0eaf5a2dfce8bea'
MODULE_PATTERN = 'eeg_all_candidates|audit_completed_finetune|qualify_.*sync|sync_qualification_boundary|qualify_deterministic_repairs'
FAULTS = {CONDITIONS[0]: '1791707600364180400-CodeBrain-lastblock-finetune-prepare', CONDITIONS[1]: '1791713045022677000-EEGMamba-lastblock-finetune-prepare'}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def desc(p):
    p = Path(p).resolve()
    require(p.is_relative_to(WORK), 'Descriptor outside workspace')
    return {'path': p.relative_to(WORK).as_posix(), 'sha256': sha(p)}

def checked(d):
    p = (WORK / d['path']).resolve()
    require(p.is_relative_to(WORK) and sha(p) == d['sha256'], 'Bound source changed: ' + d['path'])
    return p

def exclusive(path, value):
    path = Path(path)
    require(path.resolve().is_relative_to(ROOT.resolve()), 'Write outside qualification directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())

def save(path, value):
    path = Path(path)
    require(path.resolve().is_relative_to(ROOT.resolve()), 'Write outside qualification directory')
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.part')
    with temporary.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)

def check_pause(out=OUT):
    for name in ('sync_boundary', 'sync_boundary_v2', 'sync_boundary_v3'):
        path = out / 'execution_acceleration_v1' / name / 'user_pause.json'
        if path.exists():
            value = read(path)
            require(value == {'paused': False} and type(value['paused']) is bool, 'User pause or malformed pause marker blocks new work')

def check_boundary(queue=QUEUE):
    require(not (queue / 'owner.json').exists(), 'Current Q5 owner still present; do not interrupt healthy queue')
    status = read(queue / 'status.json')
    require(status.get('status') == 'required_work_remaining', 'Actual normal recovery3 terminal required')
    base = desc(queue / 'registration.json')
    core = desc(WORK / 'tridep_ccml/eeg_all_candidates/continuation_queue_v5.py')
    require(status.get('execution_version') == 5 and status.get('source') == core and (status.get('execution_registration') == base), 'Original source5/base registration required')
    require(status.get('recovery_execution_version') == 3 and status.get('recovery_registration') == desc(queue / 'recovery_registration_v3.json') and (status['recovery_registration']['sha256'] == MANIFEST_SHA), 'Canonical recovery3 terminal required')
    require(status.get('recovery_execution_source') == desc(WORK / 'tridep_ccml/eeg_all_candidates/recover_q5_after_process_loss_v3.py'), 'Actual recovery3 source differs')
    closure = status.get('recovery_closure')
    require(type(closure) is dict and closure == desc(OUT / 'process_loss_closures/lost_f3_v3_1791767849813215500.json'), 'Canonical loss3 closure required')
    loss = read(checked(closure))
    require(loss.get('recovery_registration') == status['recovery_registration'] and loss.get('closure_source') == status['recovery_execution_source'], 'Loss3/source/manifest binding differs')
    target = 'LaBraM-native4s-lastblock-finetune-fit'
    require(status.get('outcomes') == {target: 'complete'} and status.get('actual_resumed_job_ids') == [target] and (status.get('other_jobs_executed_this_recovery') == 0), 'Actual original target-only normal dispatch required')
    attempt = read(queue / 'attempts/1791767857117600800-LaBraM-native4s-lastblock-finetune-fit/status.json')
    original = next((j for j in read(queue / 'registration.json')['jobs'] if j['id'] == target))
    require(attempt.get('job') == original and original.get('fits') == 15 and (attempt.get('status') == 'complete') and (type(attempt.get('returncode')) is int) and (attempt['returncode'] == 0), 'Original complete15 child with actual exit0 required')
    require(attempt.get('source') == core and attempt.get('registration') == base and (attempt.get('execution_registration') == base) and (attempt.get('execution_version') == 5), 'Original child source/base identity differs')
    require(all((attempt.get(k) == status[k] for k in ('recovery_registration', 'recovery_execution_source', 'recovery_closure', 'recovery_execution_version'))), 'Actual child recovery3 lineage differs')
    checked(attempt['log'])
    registry = read(OUT / 'registration.json')
    others = [f for f in registry['fits'] if f['condition'] not in CONDITIONS]
    require(len(others) == 330 and len({f['fit_id'] for f in others}) == 330, 'Original nonrepair330 unique identities required')
    for fit in others:
        value = read(OUT / 'runs' / fit['fit_id'] / 'status.json')
        require(value.get('status') == 'training_complete' and value.get('fit_id') == fit['fit_id'] and (value.get('registration') == desc(OUT / 'registration.json')) and (value.get('evidence', {}).get('passed') is True), 'Nonrepair completion identity/evidence missing')
        require(read(checked(value['event_record'])) == {k: v for k, v in value.items() if k != 'event_record'}, 'Nonrepair completion differs from immutable event')
    return status

def source_check():
    records = [desc(SOURCE)]
    for relative, digest in PINS.items():
        record = desc(WORK / relative)
        require(record['sha256'] == digest, 'Qualification prototype changed')
        records.append(record)
    manifest = QUEUE / 'recovery_registration_v3.json'
    require(sha(manifest) == MANIFEST_SHA and sha(OUT / 'registration.json') == MATRIX_SHA, 'Recovery/matrix original bytes changed')
    value = read(manifest)
    sources = value.get('execution_sources', value.get('sources', value.get('experimental_sources')))
    require(isinstance(sources, list) and len(sources) == 88, 'Actual88-source recovery binding required')
    for item in sources:
        checked(item)
    records.extend([desc(manifest), desc(OUT / 'registration.json')])
    return records

def ps(code):
    executable = 'powershell.exe' if sys.platform == 'win32' else '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
    encoded = base64.b64encode(code.encode('utf-16le')).decode('ascii')
    result = subprocess.run([executable, '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded], capture_output=True, text=True, check=True, timeout=40)
    return result.stdout.lstrip('\ufeff').strip()

def host_inventory():
    code = "$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '" + MODULE_PATTERN + "' -and $_.Name -match 'python|wsl' } | ForEach-Object { $p=Get-Process -Id $_.ProcessId -ErrorAction Stop; [pscustomobject]@{pid=$_.ProcessId;parent=$_.ParentProcessId;birth=$p.StartTime.ToUniversalTime().ToFileTimeUtc().ToString();command=$_.CommandLine} }) | ConvertTo-Json -Depth 4 -Compress"
    raw = ps(code)
    items = json.loads(raw) if raw else []
    if isinstance(items, dict):
        items = [items]
    return items

def admit_host(items, allowed):
    require(isinstance(items, list) and isinstance(allowed, list), 'Native inventory unavailable')
    require(all((role in items for role in allowed)), 'A captured controller/worker role disappeared or changed')
    known = {(x['pid'], x['birth']) for x in allowed}
    require(all(((x['pid'], x['birth']) in known for x in items)), 'Another experimental native process remains')

def own_host(items):
    require(sys.platform == 'win32', 'Controller requires Windows')
    own = next((x for x in items if x['pid'] == os.getpid()), None)
    require(own is not None and 'qualify_deterministic_repairs_v7' in own['command'], 'Own native identity absent')
    allowed = [own]
    parent = next((x for x in items if x['pid'] == own['parent']), None)
    if parent:
        require(parent['command'] == own['command'], 'Unknown experimental parent')
        allowed.append(parent)
    return allowed

def linux_inventory():
    result = subprocess.run(['wsl.exe', '-d', 'TRI-DEP-Ubuntu-24.04', '--', 'ps', '-eo', 'pid,ppid,args'], capture_output=True, text=True, check=True, timeout=40)
    import re
    return [line for line in result.stdout.splitlines() if re.search(MODULE_PATTERN, line) and (not re.search('\\bps -eo\\b', line))]

def gpu_empty():
    if sys.platform != 'win32':
        output = ps("$ErrorActionPreference='Stop'; & nvidia-smi.exe -q -x; if($LASTEXITCODE -ne 0){throw 'GPU query failed'}")
    else:
        output = subprocess.run(['nvidia-smi.exe', '-q', '-x'], capture_output=True, text=True, check=True, timeout=30).stdout
    tree = ET.fromstring(output)
    gpus = tree.findall('gpu')
    require(len(gpus) == 1, 'Exactly one GPU required')
    processes = gpus[0].find('processes')
    require(processes is not None, 'GPU process inventory missing')
    require(not processes.findall('process_info'), 'GPU has an existing context; qualification deferred')
    return {'xml_sha256': hashlib.sha256(output.encode()).hexdigest(), 'context_count': 0, 'gpu_count': 1, 'captured_at': now()}

def safe_before_model(condition):
    require(condition == CONDITIONS[1], 'CodeBrain already qualified in v5; reuse successful report, do not rerun')
    require(all(not (OUT / f'deterministic_repair_qualification_v{version}/owner.json').exists() for version in (5, 6)), 'Previous qualifier lease not cleared')
    check_pause()
    terminal = check_boundary()
    sources = source_check()
    require(not (ROOT / 'owner.json').exists(), 'Repair qualification lease still present')
    require(not (OUT / 'execution_acceleration_v1/sync_qualification_owner.json').exists(), 'Optional qualification lease remains')
    fault = QUEUE / 'attempts' / FAULTS[condition] / 'status.json'
    require(read(fault).get('status') == 'failed', 'Original failed engineering evidence missing')
    matrix = read(OUT / 'registration.json')
    fits = [x for x in matrix['fits'] if x['condition'] == condition]
    require(len(fits) == 15 and all((read(OUT / 'runs' / x['fit_id'] / 'status.json').get('status') == 'blocked' for x in fits)), 'Qualification only for the original15 blocked fits; never rerun completed training')
    host = host_inventory()
    allowed = own_host(host)
    admit_host(host, allowed)
    require(linux_inventory() == [], 'Experimental Linux process remains')
    gpu = gpu_empty()
    completion = verify_original330_complete()
    return {'source': desc(SOURCE), 'sources': sources, 'terminal': desc(QUEUE / 'status.json'), 'terminal_status': terminal, 'fault_status': desc(fault), 'host_inventory': host, 'allowed_host': allowed, 'GPU_admission': gpu, 'condition': condition, 'original330_complete_proof': completion}

def validate_worker(admission, condition, admission_sha256=None):
    if sys.platform == 'linux':
        require(os.getuid() == 1000, 'Model worker must remain original unprivileged uid1000; root is read-only observer only')
    require(admission.get('condition') == condition and admission.get('source') == desc(SOURCE), 'Worker source/condition differs from admission')
    require(admission['terminal'] == desc(QUEUE / 'status.json'), 'Q5 terminal changed')
    check_pause()
    check_boundary()
    source_check()
    for item in admission['sources']:
        checked(item)
    owner = read(ROOT / 'owner.json')
    if sys.platform == 'linux':
        deadline = time.monotonic() + 20
        while not (owner.get('linux_worker_identity') and owner.get('child_native_roles') and owner.get('child_pid')):
            require(time.monotonic() < deadline, 'Controller has not captured actual Linux worker')
            time.sleep(0.1)
            owner = read(ROOT / 'owner.json')
    require(owner['condition'] == condition and owner['native_roles'] == admission['allowed_host'], 'Exact controller lease changed')
    items = host_inventory()
    allowed = list(admission['allowed_host'])
    if sys.platform == 'win32':
        own = next((x for x in items if x['pid'] == os.getpid()), None)
        require(own is not None and '--worker' in own['command'] and ('qualify_deterministic_repairs_v7' in own['command']) and (admission['condition'] in own['command']), 'Worker native identity missing')
        allowed.append(own)
        parent = next((x for x in items if x['pid'] == own['parent']), None)
        if parent and (parent['pid'], parent['birth']) not in {(x['pid'], x['birth']) for x in allowed}:
            require(parent['command'] == own['command'] and parent['parent'] == owner['pid'], 'Unexpected native worker launcher')
            allowed.append(parent)
        else:
            require(own['parent'] == owner['pid'], 'Worker parent differs from controller')
    else:
        require(sys.platform == 'linux' and admission_sha256, 'Exact Linux admission required')
        tree = N.verify(items, owner['child_native_roles'], launcher_pid=owner['child_pid'], controller_pid=owner['pid'], module='tridep_ccml.eeg_all_candidates.qualify_deterministic_repairs_v7', nonce=admission_sha256)
        require('wsl.exe' in tree[0]['command'].lower(), 'Exact WSL Popen launcher required')
        allowed.extend(tree)
    admit_host(items, allowed)
    if admission_sha256 is not None:
        expected = [x for x in allowed if x not in admission['allowed_host']]
        require(expected and all((admission_sha256 in x['command'] for x in expected)), 'Worker native command does not bind the exact admission bytes')
    linux_binding = None
    if sys.platform == 'linux':
        linux_binding = L.validate_bound(L.snapshot(), owner['linux_worker_identity'], module='tridep_ccml.eeg_all_candidates.qualify_deterministic_repairs_v7', flags={'--condition': condition, '--admission-sha256': admission_sha256}, worker_pid=os.getpid())
    return {'host_inventory': items, 'allowed_host': allowed, 'controller_lease': desc(ROOT / 'owner.json'), 'linux_worker_identity': linux_binding}

def run_controller(condition):
    admission = safe_before_model(condition)
    folder = ROOT / condition / str(time.time_ns())
    record = {'condition': condition, 'kind': 'deterministic_repair_engineering', 'formal_fits': 0, 'formal_optimizer_updates': 0, 'status': 'admitted', 'started_at': now(), 'admission': admission, 'accumulation_order_changed': condition == CONDITIONS[1], 'new_formal_execution_required': True}
    intent = ROOT / condition / 'permanent_intent.json'
    exclusive(intent, {'source': desc(SOURCE), 'folder': folder.relative_to(WORK).as_posix(), 'condition': condition, 'created_at': now()})
    exclusive(ROOT / 'owner.json', {'pid': os.getpid(), 'native_roles': admission['allowed_host'], 'condition': condition, 'intent': desc(intent)})
    try:
        exclusive(folder / 'admission.json', admission)
        args = ['-B', '-u', '-m', 'tridep_ccml.eeg_all_candidates.qualify_deterministic_repairs_v7', '--worker', '--condition', condition, '--admission', str(folder / 'admission.json'), '--admission-sha256', sha(folder / 'admission.json')]
        if condition == CONDITIONS[1]:

            def linux_path(p):
                p = Path(p).resolve()
                return '/mnt/' + p.drive[0].lower() + '/' + p.as_posix()[3:]
            arguments = [linux_path(WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/assets/eegmamba_linux_env/bin/python')] + args
            arguments[arguments.index('--admission') + 1] = linux_path(folder / 'admission.json')
            command = ['wsl.exe', '-d', 'TRI-DEP-Ubuntu-24.04', '--cd', linux_path(WORK), '--', 'env', 'CUBLAS_WORKSPACE_CONFIG=:4096:8', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'PYTHONDONTWRITEBYTECODE=1'] + arguments
        else:
            command = [str(WORK / 'S00/.venv/Scripts/python.exe')] + args
        started = time.monotonic()
        record.update(status='worker_starting')
        exclusive(folder / 'controller.json', record)
        with (folder / 'process.log').open('x', encoding='utf-8') as log:
            child = subprocess.Popen(command, cwd=WORK, stdout=log, stderr=subprocess.STDOUT, creationflags=134217728)
            record['child_pid'] = child.pid
            save(folder / 'controller.json', record)
            if condition == CONDITIONS[1]:
                current = read(ROOT / 'owner.json')
                linux_identity = L.wait_capture(module='tridep_ccml.eeg_all_candidates.qualify_deterministic_repairs_v7', flags={'--condition': condition, '--admission-sha256': sha(folder / 'admission.json')}, is_alive=lambda: child.poll() is None)
                items = host_inventory()
                native_tree = N.capture(items, launcher_pid=child.pid, controller_pid=os.getpid(), module='tridep_ccml.eeg_all_candidates.qualify_deterministic_repairs_v7', nonce=sha(folder / 'admission.json'))
                admit_host(items, current['native_roles'] + native_tree)
                current.update(linux_worker_identity=linux_identity, child_pid=child.pid, child_native_roles=native_tree)
                save(ROOT / 'owner.json', current)
            code = child.wait()
        record.update(returncode=code, status='worker_returned', finished_at=now(), inclusive_controller_seconds=time.monotonic() - started, controller_seconds_primary_cost=False)
        save(folder / 'controller.json', record)
        require(code == 0 and read(folder / 'report.json')['status'] == 'engineering_verified', 'Actual worker failed or report unverified; retain intent and inspect')
        return desc(folder / 'report.json')
    finally:
        if 'code' in locals() or 'child' not in locals():
            owner = read(ROOT / 'owner.json')
            require(owner['pid'] == os.getpid() and owner['intent'] == desc(intent), 'Exact qualification owner changed')
            exclusive(folder / 'released_owner.json', owner)
            (ROOT / 'owner.json').unlink()

def isolated_builder(F):
    import importlib
    mapping = {'.adapters.codebrain_v2': '.adapters.codebrain_v3', '.adapters.eegmamba': '.adapters.eegmamba_v2'}

    def route(name, package=None):
        return importlib.import_module(mapping.get(name, name), package)
    original = F.build_model
    private = dict(original.__globals__)
    private['importlib'] = types.SimpleNamespace(import_module=route)
    result = types.FunctionType(original.__code__, private, original.__name__, original.__defaults__, original.__closure__)
    result.__kwdefaults__ = original.__kwdefaults__.copy()
    require(result.__code__ is original.__code__ and F.build_model is original, 'Original builder changed')
    return result

def runtime_worker(path, digest, condition):
    require(sha(path) == digest, 'Admission bytes differ')
    admission = read(path)
    folder = Path(path).parent
    require(folder.resolve().is_relative_to(ROOT.resolve()), 'Worker directory outside qualification')
    exclusive(folder / 'worker_claim.json', {'pid': os.getpid(), 'platform': sys.platform, 'source': desc(SOURCE), 'started_at': now()})
    worker_native = validate_worker(admission, condition, admission_sha256=digest)
    worker_gpu_admission = gpu_empty()
    if condition == CONDITIONS[1]:
        expected = OUT / 'assets/eegmamba_linux_env'
        require(sys.platform == 'linux' and Path(sys.prefix).resolve() == expected.resolve(), 'Pinned official Linux runtime required')
    else:
        require(sys.platform == 'win32', 'CodeBrain requires pinned Windows runtime')
    exclusive(folder / 'worker_owner.json', {'pid': os.getpid(), 'parent': os.getppid(), 'platform': sys.platform, 'started_at': now(), 'native': worker_native, 'GPU_before_runtime': worker_gpu_admission, 'linux_start_ticks': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19] if sys.platform == 'linux' else None})
    record = {'condition': condition, 'kind': 'deterministic_repair_engineering', 'started_at': now(), 'status': 'running', 'source': desc(SOURCE), 'admission': desc(path), 'formal_fits': 0, 'optimizer_step_started': 0, 'optimizer_step_returned': 0, 'forward_started': 0, 'forward_returned': 0, 'backward_started': 0, 'backward_returned': 0, 'encoder_forward_started': 0, 'encoder_forward_returned': 0, 'encoder_windows_started': 0, 'encoder_windows_returned': 0, 'outer_test_read': False, 'native_CUDA_gradient_equivalence_established': False, 'performance_improvement_claimed': False, 'accumulation_order_changed': condition == CONDITIONS[1], 'new_formal_execution_required': True}
    exclusive(folder / 'attempt.json', record)
    started = time.monotonic()
    model = optimizer = raw = None
    F = torch = None

    def observed(kind, fn, window=False):

        def call(*args, **kwargs):
            check_pause()
            require(not (QUEUE / 'owner.json').exists() and desc(ROOT / 'owner.json') == worker_native['controller_lease'], 'Queue/qualification ownership changed')
            admit_host(host_inventory(), worker_native['allowed_host'])
            if sys.platform == 'linux':
                L.validate_bound(L.snapshot(), worker_native['linux_worker_identity'], module='tridep_ccml.eeg_all_candidates.qualify_deterministic_repairs_v7', flags={'--condition': condition, '--admission-sha256': digest}, worker_pid=os.getpid())
            record[kind + '_started'] += 1
            if window:
                record['encoder_windows_started'] += len(args[1])
            save(folder / 'attempt.json', record)
            result = fn(*args, **kwargs)
            record[kind + '_returned'] += 1
            if window:
                record['encoder_windows_returned'] += len(args[1])
            save(folder / 'attempt.json', record)
            return result
        return call
    try:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        from . import finetune as F
        torch = F.torch
        if condition == CONDITIONS[1]:
            import importlib.metadata
            require(torch.__version__ == '2.5.1+cu124' and importlib.metadata.version('mamba_ssm') == '2.2.4', 'Linux runtime versions differ')
        F.C.runtime(11)
        require(torch.are_deterministic_algorithms_enabled() and (not torch.is_deterministic_algorithms_warn_only_enabled()), 'Strict determinism must remain enabled')
        _, folds, labels = F.C.context()
        split = folds['fold_1']
        person = split['train'][0]
        require(person not in split['val'] + split['test'], 'Training-only qualification required')
        raw, raw_sources = F._raw_sources(split, people=[person])
        model, audit = isolated_builder(F)(condition, seed=11, device='cuda', window_chunk=8)
        require(model.window_chunk == 8 and len(model.selected_names) > 0, 'Original chunk8/lastblock list required')
        initial = F.state_digest(model.state_dict())
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        model.forward = observed('forward', model.forward)
        namespace = dict(vars(model.adapter))
        namespace['encode_windows'] = observed('encoder_forward', model.adapter.encode_windows, True)
        model.adapter = types.SimpleNamespace(**namespace)
        optimizer = F.optimizer_for(model)

        def forbid_step(*args, **kwargs):
            record['optimizer_step_started'] += 1
            save(folder / 'attempt.json', record)
            raise ValueError('Zero optimizer qualification: step prohibited')
        optimizer.step = forbid_step
        blocks = [b.S41 for b in model.encoder.residual_layer.residual_blocks] if condition == CONDITIONS[0] else [b.mixer for b in model.encoder.encoder.layers]
        private = [b.forward for b in blocks]
        model.eval()
        batch = F.np.stack([raw[person][F.bag_indices(1)]])
        rng = F.rng_state(model.device)
        try:
            for b in blocks:
                b.forward = types.MethodType(type(b).forward, b)
            with torch.no_grad():
                native = model(batch).detach().cpu()
        finally:
            for b, fn in zip(blocks, private):
                b.forward = fn
        F.restore_rng(rng)
        with torch.no_grad():
            adapted = model(batch).detach().cpu()
        require(torch.equal(native, adapted) and F.rng_equal(F.rng_state(model.device), rng), 'Original/private forward or RNG differ')
        phases = []
        for epoch in (1, 11):
            enabled = F.set_train_phase(model, epoch)
            rng = F.rng_state(model.device)
            captures = []
            for repeat in range(2):
                check_pause()
                F.restore_rng(rng)
                optimizer.zero_grad(set_to_none=True)
                target = torch.tensor([labels[person]], device=model.device, dtype=torch.long)
                batch = F.np.stack([raw[person][F.bag_indices(epoch)]])
                loss = torch.nn.functional.cross_entropy(model(batch), target)
                require(bool(torch.isfinite(loss)), 'Nonfinite engineering loss')
                observed('backward', loss.backward)()
                gradients = {}
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        require(p.grad is not None and bool(torch.isfinite(p.grad).all()), 'Missing/nonfinite gradient: ' + name)
                        gradients[name] = p.grad.detach().cpu().clone()
                    else:
                        require(p.grad is None, 'Frozen parameter acquired gradient: ' + name)
                captures.append({'loss': loss.detach().cpu().clone(), 'gradients': gradients, 'rng': F.rng_state(model.device)})
            a, b = captures
            require(torch.equal(a['loss'], b['loss']) and a['gradients'].keys() == b['gradients'].keys() and all((torch.equal(a['gradients'][n], b['gradients'][n]) for n in a['gradients'])) and F.rng_equal(a['rng'], b['rng']), 'Repeated strict GPU loss/gradients/RNG differ')
            phases.append({'epoch_phase': epoch, 'enabled_subtrees': enabled, 'gradient_parameter_names': list(a['gradients']), 'gradient_state_sha256': F.state_digest(a['gradients']), 'repeat_bitwise_equal': True})
        require(F.state_digest(model.state_dict()) == initial, 'Zero-update probe changed parameter/buffer bytes')
        optimizer.zero_grad(set_to_none=True)
        state = F.rng_state(model.device)
        checkpoint = folder / 'resume_probe.pt'
        value = {'state_dict': F.cpu_state(model.state_dict()), 'optimizer_state_dict': F.cpu_state(optimizer.state_dict()), 'scheduler_state_dict': {'policy': 'finetune', 'completed_epoch': 0, 'extended': False}, 'rng': state, 'sampler_position': {'epoch_completed': 0, 'next_batch': 0}}
        require(not checkpoint.exists(), 'No engineering checkpoint overwrite')
        torch.save(value, checkpoint)
        with checkpoint.open('r+b') as f:
            os.fsync(f.fileno())
        restored = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(restored['state_dict'], strict=True)
        optimizer.load_state_dict(restored['optimizer_state_dict'])
        F.restore_rng(restored['rng'])
        require(F.state_digest(model.state_dict()) == initial and F.rng_equal(state, F.rng_state(model.device)) and (optimizer.state_dict()['state'] == {}) and (restored['sampler_position'] == value['sampler_position']) and (restored['scheduler_state_dict'] == value['scheduler_state_dict']), 'Complete zero-step engineering resume differs')
        require(record['forward_started'] == record['forward_returned'] == 6 and record['backward_started'] == record['backward_returned'] == 4 and (record['encoder_windows_started'] == record['encoder_windows_returned'] == 48) and (record['optimizer_step_started'] == record['optimizer_step_returned'] == 0), 'Observed engineering counts differ')

        def descriptors(value):
            if isinstance(value, dict):
                if 'path' in value and 'sha256' in value:
                    yield {key: value[key] for key in ('path', 'sha256')}
                else:
                    for child in value.values():
                        yield from descriptors(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    yield from descriptors(child)
        resources = {x['path']: x for x in descriptors(audit)}
        require(resources, 'Actual author weight/source descriptors required')
        for resource in resources.values():
            checked(resource)
        implementation = F._implementation_sources(condition, model, [x for x in resources.values() if x['path'].endswith('.py')])
        implementation.append(desc(SOURCE))
        update_names = list(model.update_names)
        require(set(update_names) == set(phases[1]['gradient_parameter_names']), 'Qualified lastblock gradients differ from exact update names')
        runtime_contract = {'input_contract': {'adapter': audit, 'windows_per_training_subject': 8, 'all_window_validation': 60, 'coverage_cycle_epochs': 8, 'micro_subjects': 1, 'window_chunk': 8, 'effective_subjects': 8, 'precision': 'float32'}, 'trainable_parameter_names': update_names, 'frozen_parameter_names': [name for name, _ in model.named_parameters() if name not in update_names], 'implementation_sources': implementation, 'weight_sources': list(resources.values()), 'forward_checked': True, 'gradient_checked': True, 'resume_checked': True, 'training_only_engineering_checked': True, 'probe_subject_ids': [person], 'raw_sources': raw_sources, 'engineering_optimizer_updates': 0, 'engineering_backward_calls': record['backward_returned'], 'qualification_source': desc(SOURCE), 'frozen_buffers_and_parameters_zero_update_verified': True, 'native_CUDA_gradient_equivalence_established': False, 'accumulation_order_changed': condition == CONDITIONS[1], 'separate_new_formal_execution_and_auditor_required': True}
        record.update(status='engineering_verified', runtime_contract=runtime_contract, GPU_engineering_checks_passed=True, checks=phases, adapter_audit=audit, probe_person=person, raw_sources=raw_sources, original_native_forward_bitwise_equal=True, initial_state_sha256=initial, resume_probe=desc(checkpoint), zero_step_optimizer_and_rng_resume_verified=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(), baseline_allocated_bytes=baseline, host_transfer_cost_separately_instrumented=False, elapsed_scope='includes torch import/runtime, constructor, raw load, forward/backward, hashing, host transfers and checkpoint; no speed comparison')
    except BaseException as error:
        record.update(status='failed', failure_type=type(error).__name__, failure=str(error), unreturned_call_internal_work_unknown=True)
        raise
    finally:
        record.update(finished_at=now(), primary_engineering_seconds=time.monotonic() - started)
        save(folder / 'attempt.json', record)
        if record['status'] == 'engineering_verified':
            exclusive(folder / 'report.json', record)
        del model, optimizer, raw
        import gc
        gc.collect()
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.empty_cache()

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--condition', choices=CONDITIONS)
    parser.add_argument('--admission')
    parser.add_argument('--admission-sha256')
    args = parser.parse_args(argv)
    require(not (args.run and args.worker), 'Choose controller or worker')
    if not args.run and (not args.worker):
        require(args.admission is None and args.admission_sha256 is None, 'Admission only for actual worker')
        print(json.dumps({'status': 'preparation_only', 'model_calls': 0, 'GPU_queries': 0, 'file_reads': 0, 'writes': 0, 'optimizer_steps': 0, 'formal_fits': 0, 'GPU_qualification_complete': False, 'new_formal_execution_required': True, 'conditions': list(CONDITIONS)}))
        return
    require(args.condition is not None, 'An explicit blocked condition required')
    if args.worker:
        require(args.admission is not None and args.admission_sha256 is not None, 'Exact worker admission required')
        return runtime_worker(Path(args.admission), args.admission_sha256, args.condition)
    require(args.admission is None and args.admission_sha256 is None, 'Controller creates its own actual admission')
    print(json.dumps(run_controller(args.condition)))

def verify_original330_complete():
    from . import registry as R
    matrix = R.load_registration()
    rows = []
    for fit in matrix['fits']:
        if fit['condition'] in CONDITIONS:
            continue
        folder = R.fit_directory(fit['condition'], fit['seed'], fit['fold'])
        value = R.fit_status(fit['condition'], fit['seed'], fit['fold'])
        require(value['status'] == 'training_complete', 'Original330 fitting incomplete')
        R._validate_complete(value['evidence'], folder)
        rows.append({'fit_id': fit['fit_id'], 'status': desc(folder / 'status.json'), 'complete': desc(folder / 'complete.json'), 'evidence': value['evidence']})
    require(len(rows) == 330 and len({r['fit_id'] for r in rows}) == 330, 'Actual full330 completion proof required')
    return {'source': desc(SOURCE), 'registry': desc(OUT / 'registration.json'), 'recovery_registration': desc(QUEUE / 'recovery_registration_v3.json'), 'formal_fits_verified': 330, 'rows': rows, 'checkpoint_tensor_loads': 0, 'outer_probability_reads': 0}
if __name__ == '__main__':
    main()
