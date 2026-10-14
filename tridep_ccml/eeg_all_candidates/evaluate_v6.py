"""Owned E6 evaluation of true old/new training; default zero operation.

Actual360 plus the actual CPU10 135/405 gate and released Q5/Q8 owners precede
any model/raw/probability read. Linux Mamba staged predictions run first, then
Windows all23/435 aggregation. Original metrics/thresholds and E2 selection
disclosure are preserved. Private repaired backward is not used for inference.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types
import uuid
from . import storage as S, protocol as P, registry as R
from . import evaluate_v2 as E2, execution_acceleration_v1 as X
from . import repair_observation_v3 as V
from . import qualify_deterministic_repairs_v5 as Q
from . import linux_worker_identity_v1 as L
SOURCE = Path(__file__).resolve()
ROOT = E2.ROOT
CONTROL = S.OUT / 'evaluation_v6_control'
CONDITIONS = E2.CONDITIONS
MODULE = 'tridep_ccml.eeg_all_candidates.evaluate_v6'
selection_evidence = E2.selection_evidence

def require(ok, message):
    if not ok:
        raise ValueError(message)

def execution_provenance(condition=None):
    value = E2.execution_provenance(condition)
    return {**value, 'version': 6, 'wrapper_source': S.desc(SOURCE), 'delegated_evaluator_wrapper_source': S.desc(Path(E2.__file__)), 'previous_E3_source': S.desc(SOURCE.with_name('evaluate_v3.py')), 'old_execution_verifier_source': S.desc(Path(X.__file__)), 'new_repair_execution_verifier_source': S.desc(V.SOURCE), 'old_and_repair6_training_provenance_verified': True, 'original_native_forward_reused_no_backward_for_test': True, 'new_backward_used_or_optimized_during_test': False}

def verify_execution(value, condition=None):
    require(value == execution_provenance(condition), 'Actual E6 inference source differs')
    for item in value.values():
        if isinstance(item, dict) and set(item) == {'path', 'sha256'}:
            S.checked(item)

def verify_training_execution(metadata):
    if 'actual_deterministic_repair_execution' in metadata:
        return V.verify_execution(metadata, condition=metadata['condition'], seed=metadata['seed'], fold=metadata['fold'])
    if 'execution_acceleration' in metadata:
        return X.verify_execution_acceleration(metadata, condition=metadata['condition'], seed=metadata['seed'], fold=metadata['fold'])
    return None

def isolated_driver():
    wrapper = types.ModuleType(__package__ + '._evaluation_v4_private_E2')
    wrapper.__package__ = __package__
    wrapper.__file__ = E2.__file__
    exec(compile(Path(E2.__file__).read_text(encoding='utf-8'), E2.__file__, 'exec'), wrapper.__dict__)
    wrapper.SOURCE = SOURCE
    wrapper.execution_provenance = execution_provenance
    wrapper.verify_execution = verify_execution
    previous_materialize = wrapper.materialize_finetune

    def materialize(*args, **kwargs):
        metadata = kwargs.get('metadata', args[4] if len(args) > 4 else None)
        verify_training_execution(metadata)
        return previous_materialize(*args, **kwargs)
    wrapper.materialize_finetune = materialize
    driver = wrapper.isolated_driver()
    previous_context = driver._fit_context

    def context(condition, *args, **kwargs):
        metadata, plan, binding, destination = previous_context(condition, *args, **kwargs)
        if P.CONDITION_BY_ID[condition].policy_key == 'finetune':
            proof = verify_training_execution(metadata)
            if proof is not None:
                binding['training_execution_acceleration'] = proof
                binding['actual_training_observer_version'] = proof.get('observer_version', proof.get('version'))
        return (metadata, plan, binding, destination)
    driver._fit_context = context
    return driver

def validate_gate_data(gate, cpu, cpu_source):
    require(gate.get('passed') is True and gate.get('original_training_complete') == 360 and (gate.get('FT') == 135) and (gate.get('CPU_loads') == 405), 'Actual360 and135/405 prerequisite required')
    require(cpu.get('schema_version') == 10 and cpu.get('auditor_source') == cpu_source and (cpu.get('all_135_finetune_checkpoints_verified') is True) and (cpu.get('verified_completed_finetune_fits') == 135) and (cpu.get('training_complete_finetune_observed') == 135) and (cpu.get('registered_finetune_fits') == 135) and (cpu.get('checkpoint_loads') == 405) and (cpu.get('errors') == []) and (cpu.get('deferred_completed_fits') == []) and (cpu.get('pending_fits') == []), 'Actual full CPU10 report without errors/deferred/pending required')

def verify_gate(binding):
    Q.check_pause()
    require(not (S.OUT / 'continuation_queue_v5/owner.json').exists() and (not (S.OUT / 'continuation_queue_v8/owner.json').exists()), 'Training queue owner still present')
    require(R.status_summary()['statuses'].get('training_complete') == 360, 'All360 actual formal fits still required')
    path = S.checked(binding)
    require(path.parent == (S.OUT / 'continuation_queue_v8').resolve() and path.name.startswith('pre_test_CPU_gate_'), 'Canonical actual Q8 CPU gate required')
    gate = S.read(path)
    require(gate.get('source') == S.desc(SOURCE.with_name('continuation_queue_v8.py')) and gate.get('registration') == S.desc(S.OUT / 'continuation_queue_v8/registration.json'), 'Current actual Q8 gate source differs')
    cpu = S.read(S.checked(gate['CPU_audit']))
    validate_gate_data(gate, cpu, S.desc(S.REPORT / 'audit_completed_finetune_checkpoints_v10.py'))
    queue = S.read(S.checked(gate['registration']))
    require(queue['version'] == 8 and queue['schema'] == 'eeg_deterministic_repair_queue_v8', 'Actual source7 queue registration required')
    for item in queue['sources']:
        S.checked(item)
    require(S.desc(SOURCE) in queue['sources'], 'Actual E6 source not registered')
    return gate

def create(path, value):
    path = Path(path)
    require(path.resolve().is_relative_to(CONTROL.resolve()), 'Owned evaluation output outside control directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    return S.desc(path)

def native_roles(items, pid):
    actual = next((x for x in items if x['pid'] == pid), None)
    require(actual is not None and MODULE in actual['command'], 'Actual evaluator native identity absent')
    roles = [actual]
    parent = next((x for x in items if x['pid'] == actual['parent']), None)
    if parent:
        require(parent['command'] == actual['command'], 'Unknown evaluator parent')
        roles.append(parent)
    return roles

def admit_worker(items, owner, token, worker_pid, platform):
    require(owner['source'] == S.desc(SOURCE) and owner.get('child_token') == token, 'Actual E6 owner/source/nonce differs')
    roles = list(owner['native_roles'])
    for role in roles:
        require(role in items, 'Actual evaluator parent birth/command changed')
    launch = next((x for x in items if x['pid'] == owner['child_pid']), None)
    require(launch is not None and launch == owner['child_identity'] and (launch['parent'] == owner['pid']) and ('--worker' in launch['command']) and (token in launch['command']), 'Actual evaluator child launcher differs')
    roles.append(launch)
    if platform == 'win32':
        actual = next((x for x in items if x['pid'] == worker_pid), None)
        require(actual is not None, 'Actual evaluator Python identity absent')
        if actual['pid'] != launch['pid']:
            require(actual['parent'] == launch['pid'] and actual['command'] == launch['command'], 'Evaluator Python child outside launcher tree')
            roles.append(actual)
    else:
        require(platform == 'linux' and 'wsl.exe' in launch['command'].lower(), 'Original WSL launcher required')
        require(owner.get('linux_worker_identity') is not None, 'Controller-captured evaluator Linux identity required')
        L.validate_bound(L.snapshot(), owner['linux_worker_identity'], module=MODULE, flags={'--token': token, '--mode': owner['child_mode'], '--admission-sha256': owner['child_admission']['sha256'], '--cpu-gate-sha256': owner['CPU_gate']['sha256']}, worker_pid=worker_pid)
    Q.admit_host(items, roles)
    return roles

def worker(*, gate_binding, admission_binding, token, mode):
    verify_gate(gate_binding)
    admission = S.read(S.checked(admission_binding))
    require(admission['source'] == S.desc(SOURCE) and admission['token'] == token and (admission['CPU_gate'] == gate_binding) and (admission['mode'] == mode), 'Bound evaluator admission differs')
    owner_path = CONTROL / 'owner.json'
    deadline = time.monotonic() + 20
    while True:
        Q.check_pause()
        owner = S.read(owner_path)
        if owner.get('child_identity') and owner.get('child_token') == token and (sys.platform != 'linux' or owner.get('linux_worker_identity')):
            break
        require(time.monotonic() < deadline, 'Actual child birth not published')
        time.sleep(0.1)
    require(owner['id'] == admission['queue_owner_id'], 'Actual E6 owner belongs to another admission')
    roles = admit_worker(Q.host_inventory(), owner, token, os.getpid(), sys.platform)
    GPU = Q.gpu_empty()
    if mode == 'mamba_linux':
        require(sys.platform == 'linux' and Path(sys.prefix).resolve() == (S.OUT / 'assets/eegmamba_linux_env').resolve(), 'Original qualified Linux Mamba environment required')
        conditions = ('EEGMamba-lastblock-finetune',)
        no_report = True
    else:
        require(mode == 'all_windows' and sys.platform == 'win32', 'Windows aggregation mode required')
        conditions = CONDITIONS
        no_report = False
    path = S.checked(admission_binding).parent
    create(path / 'worker_owner.json', {'pid': os.getpid(), 'parent': os.getppid(), 'native_roles': roles, 'platform': sys.platform, 'linux_start_ticks': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19] if sys.platform == 'linux' else None, 'source': S.desc(SOURCE), 'CPU_gate': gate_binding, 'GPU_before_runtime': GPU, 'started_at': S.now(), 'linux_worker_identity': owner.get('linux_worker_identity')})
    record = {'status': 'running', 'source': S.desc(SOURCE), 'CPU_gate': gate_binding, 'mode': mode, 'admission': admission_binding, 'started_at': S.now(), 'formal_fits_or_optimizer_updates_added': 0}
    create(path / 'worker.json', record)
    started = time.monotonic()
    try:
        driver = isolated_driver()
        previous_fit = driver._evaluate_fit

        def owned_fit(*args, **kwargs):
            Q.check_pause()
            current = S.read(owner_path)
            require(current['id'] == admission['queue_owner_id'], 'Actual evaluator owner changed during inference')
            current_roles = admit_worker(Q.host_inventory(), current, token, os.getpid(), sys.platform)
            require(current_roles == roles, 'Evaluator actual child birth/tree changed')
            return previous_fit(*args, **kwargs)
        driver._evaluate_fit = owned_fit
        result = driver.run(conditions, no_report=no_report)
        record.update(status='complete', evaluation_result=result)
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record.update(finished_at=S.now(), inclusive_evaluator_worker_seconds=time.monotonic() - started)
        S.save(path / 'worker.json', record)

def linux_path(path):
    path = Path(path).resolve()
    require(path.drive and len(path.drive) == 2, 'Windows drive required')
    return '/mnt/' + path.drive[0].lower() + '/' + path.as_posix()[3:]

def controller(gate_binding):
    require(sys.platform == 'win32', 'Actual evaluation controller requires Windows')
    verify_gate(gate_binding)
    require(not (CONTROL / 'owner.json').exists(), 'Evaluator already owned; do not duplicate')
    items = Q.host_inventory()
    roles = native_roles(items, os.getpid())
    Q.admit_host(items, roles)
    require(Q.linux_inventory() == [], 'Another experimental Linux process remains')
    GPU = Q.gpu_empty()
    owner = {'id': uuid.uuid4().hex, 'pid': os.getpid(), 'source': S.desc(SOURCE), 'CPU_gate': gate_binding, 'native_roles': roles, 'GPU_before_queue': GPU, 'started_at': S.now(), 'child_pid': None, 'child_token': None, 'child_identity': None}
    create(CONTROL / 'owner.json', owner)
    runfolder = CONTROL / 'runs' / str(time.time_ns())
    record = {'status': 'running', 'source': S.desc(SOURCE), 'CPU_gate': gate_binding, 'started_at': S.now(), 'stages': {}, 'all_plan_complete': False}
    create(runfolder / 'status.json', record)
    try:
        for mode in ('mamba_linux', 'all_windows'):
            Q.check_pause()
            verify_gate(gate_binding)
            folder = runfolder / mode
            token = uuid.uuid4().hex
            admission = {'source': S.desc(SOURCE), 'mode': mode, 'CPU_gate': gate_binding, 'token': token, 'queue_owner_id': owner['id']}
            binding = create(folder / 'worker_admission.json', admission)
            args = ['-B', '-u', '-m', MODULE, '--worker', '--mode', mode, '--cpu-gate', gate_binding['path'], '--cpu-gate-sha256', gate_binding['sha256'], '--admission', binding['path'], '--admission-sha256', binding['sha256'], '--token', token]
            if mode == 'mamba_linux':
                command = ['wsl.exe', '-d', 'TRI-DEP-Ubuntu-24.04', '--cd', linux_path(S.WORK), '--exec', linux_path(S.OUT / 'assets/eegmamba_linux_env/bin/python')] + args
            else:
                command = [str(S.WORK / 'S00/.venv/Scripts/python.exe')] + args
            with (folder / 'process.log').open('x', encoding='utf-8') as log:
                process = subprocess.Popen(command, cwd=S.WORK, stdout=log, stderr=subprocess.STDOUT, creationflags=134217728)
                owner.update(child_pid=process.pid, child_token=token, child_identity=None, linux_worker_identity=None, child_mode=mode, child_admission=binding)
                S.save(CONTROL / 'owner.json', owner)
                deadline = time.monotonic() + 10
                while True:
                    native = next((x for x in Q.host_inventory() if x['pid'] == process.pid), None)
                    if native is not None:
                        break
                    require(process.poll() is None and time.monotonic() < deadline, 'Actual evaluator child birth unavailable')
                    time.sleep(0.1)
                require(native['parent'] == owner['pid'], 'Evaluator child outside native parent')
                owner['child_identity'] = native
                S.save(CONTROL / 'owner.json', owner)
                if mode == 'mamba_linux':
                    owner['linux_worker_identity'] = L.wait_capture(module=MODULE, flags={'--token': token, '--mode': mode, '--admission-sha256': binding['sha256'], '--cpu-gate-sha256': gate_binding['sha256']}, is_alive=lambda: process.poll() is None)
                    S.save(CONTROL / 'owner.json', owner)
                code = process.wait()
            record['stages'][mode] = {'returncode': code, 'worker_report': S.desc(folder / 'worker.json') if (folder / 'worker.json').exists() else None}
            S.save(runfolder / 'status.json', record)
            owner.update(child_pid=None, child_token=None, child_identity=None, linux_worker_identity=None, child_mode=mode, child_admission=binding)
            S.save(CONTROL / 'owner.json', owner)
            require(code == 0 and S.read(folder / 'worker.json')['status'] == 'complete', 'Actual evaluator child failed; preserve evidence')
        report = completed_report_verified()
        record.update(status='all_evaluation_complete', evaluation_proof=report, finished_at=S.now(), all_plan_complete=False)
        S.save(runfolder / 'status.json', record)
        create(CONTROL / 'complete.json', {'source': S.desc(SOURCE), 'CPU_gate': gate_binding, 'status': S.desc(runfolder / 'status.json'), 'evaluation_proof': report, 'actual_evaluation_complete': True, 'all_plan_complete': False})
        return report
    finally:
        current = S.read(CONTROL / 'owner.json')
        if current['id'] == owner['id'] and current.get('child_pid') is None:
            create(runfolder / 'released_owner.json', current)
            (CONTROL / 'owner.json').unlink()

def completed_condition_verified(condition):
    value = isolated_driver().completed_condition_verified(condition)
    value['evaluator_execution'] = execution_provenance(condition)
    return value

def completed_report_verified():
    value = isolated_driver().completed_report_verified()
    manifest = S.read(ROOT / 'complete.json')
    verify_execution(manifest.get('evaluator_execution'))
    report = S.read(S.checked(manifest['evaluation_report']))
    verify_execution(report.get('evaluator_execution'))
    value['evaluator_execution'] = execution_provenance()
    return value

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--cpu-gate')
    parser.add_argument('--cpu-gate-sha256')
    parser.add_argument('--admission')
    parser.add_argument('--admission-sha256')
    parser.add_argument('--token')
    parser.add_argument('--mode', choices=('mamba_linux', 'all_windows'))
    args = parser.parse_args(argv)
    require(not (args.run and args.worker), 'Controller and worker are distinct')
    if not args.run and (not args.worker):
        print(json.dumps({'status': 'preparation_only', 'model_calls': 0, 'GPU_queries': 0, 'outer_probability_reads': 0, 'actual_evaluation_started': False, 'report_or_owner_writes': 0}))
        return
    require(args.cpu_gate is not None and args.cpu_gate_sha256 is not None, 'Actual full CPU10 gate required')
    gate = {'path': args.cpu_gate, 'sha256': args.cpu_gate_sha256}
    if args.worker:
        require(all((args.admission, args.admission_sha256, args.token, args.mode)), 'Exact actual child admission required')
        return worker(gate_binding=gate, admission_binding={'path': args.admission, 'sha256': args.admission_sha256}, token=args.token, mode=args.mode)
    require(args.admission is None, 'Controller makes its own actual child admission')
    print(json.dumps(controller(gate)))
if __name__ == '__main__':
    main()
