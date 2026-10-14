"""Sequential new repair30 fitting and strict CPU12 gate; default no operation.

No old Q5 status/source/attempt is relabelled or restarted. This queue registers

only after both actual engineering reports and independent complete-source

review. Unified E8 evaluation is a separate owned action after this owner

releases; all360 plus actual135/405 still precede any outer-test work.

"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from . import storage as S, registry as R
from . import deterministic_repair_execution_v5 as D
from . import qualify_deterministic_repairs_v7 as Q
from . import repair_runtime_owner_v5 as O
from . import linux_worker_identity_v2 as L
from . import native_launcher_tree_v1 as N
SOURCE = Path(__file__).resolve()
ROOT = S.OUT / 'continuation_queue_v10'
SCHEMA = 'eeg_deterministic_repair_queue_v10'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def create(path, value):
    path = Path(path)
    require(path.resolve().is_relative_to(S.OUT.resolve()), 'Registration outside experiment')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return S.desc(path)

def sources():
    values = {x['path']: x for x in D.implementation_sources()}
    for name in ('repair_runtime_owner_v5.py', 'repair_owned_worker_v5.py', 'repair_observation_v5.py'):
        item = S.desc(SOURCE.with_name(name))
        values[item['path']] = item
    return list(values.values())

def original_others_complete():
    registration = R.load_registration()
    others = [x for x in registration['fits'] if x['condition'] not in D.REPAIR_CONDITIONS]
    require(len(others) == 330, 'Original non-repair330 identity scope differs')
    for fit in others:
        status = R.fit_status(fit['condition'], fit['seed'], fit['fold'])
        require(status['status'] == 'training_complete', 'Original healthy queue still has required work')
        R._validate_complete(status['evidence'], R.fit_directory(fit['condition'], fit['seed'], fit['fold']))
    return [x['fit_id'] for x in others]

def review_engineering(binding, proof):
    value = D.read_bound(binding)
    require(value.get('passed') is True and value.get('qualification_report') == proof['report'] and (value.get('controller') == proof['controller']) and (value.get('released_owner') == proof['released_owner']) and (value.get('numerical_order_change_acknowledged') is proof['contract']['accumulation_order_changed']), 'Independent actual engineering report review missing')
    S.checked(value['reviewer_source'])
    return value

def register(qualifications, reviews, source_review):
    Q.check_pause()
    Q.check_boundary()
    require(set(qualifications) == set(reviews) == set(D.REPAIR_CONDITIONS), 'Both original repair15 groups required')
    others = original_others_complete()
    all_sources = sources()
    review = D.read_bound(source_review)
    require(review.get('status') == 'new_repair_chain_sources_reviewed' and review.get('reviewed_sources') == all_sources and (review.get('passed') is True), 'Independent full current-source-chain review required; preparation review insufficient')
    S.checked(review['reviewer_source'])
    target = ROOT / 'registration.json'
    require(not target.exists() and (not (ROOT / 'registration_intent.json').exists()), 'Permanent registration already attempted; inspect actual state, never repeat transaction')
    proofs = {condition: D.qualification(qualifications[condition], condition) for condition in D.REPAIR_CONDITIONS}
    for condition in D.REPAIR_CONDITIONS:
        review_engineering(reviews[condition], proofs[condition])
        require(not (S.OUT / 'engineering' / (condition + '.json')).exists(), 'Original condition lock already exists')
        for fit_id in proofs[condition]['original_fit_ids']:
            folder = S.OUT / 'runs' / fit_id
            require(S.read(folder / 'status.json')['status'] == 'blocked' and (not (folder / 'started.json').exists()) and (not (folder / 'complete.json').exists()), 'Do not replace or relabel an existing formal trajectory')
    create(ROOT / 'registration_intent.json', {'source': S.desc(SOURCE), 'started_at': S.now(), 'source_review': source_review})
    sidecars = {}
    try:
        for condition in D.REPAIR_CONDITIONS:
            value = {'schema': D.SCHEMA, 'version': 5, 'status': 'qualified', 'condition': condition, 'registry': R.registration_descriptor(), 'sources': D.implementation_sources(), 'qualification': qualifications[condition], 'independent_engineering_review': reviews[condition], 'original_fit_ids': proofs[condition]['original_fit_ids'], 'registered_at': S.now(), 'training_rules_micro1_logical8_chunk8_FP32_unchanged': True, 'new_actual_observer_version': 8, 'original_CUDA_backward_equivalence_established': False}
            sidecars[condition] = create(D.ROOT / condition / 'registration.json', value)
            contract = D.lock_contract(proofs[condition], sidecars[condition])
            R.lock_condition(condition, contract)
            for seed in (11, 22, 33):
                for fold in (1, 2, 3, 4, 5):
                    R.record_fit_status(condition, seed, fold, 'ready')
        value = {'schema': SCHEMA, 'version': 10, 'registered_at': S.now(), 'source': S.desc(SOURCE), 'registry': R.registration_descriptor(), 'sources': all_sources, 'previous_Q5_terminal': S.desc(Q.QUEUE / 'status.json'), 'previous_recovery_registration': S.desc(Q.QUEUE / 'recovery_registration_v3.json'), 'original_other330_fit_ids': others, 'repair_registrations': sidecars, 'source_review': source_review, 'formal_fits_to_run': 30, 'original_fit_ids': [x for c in D.REPAIR_CONDITIONS for x in proofs[c]['original_fit_ids']], 'single_gpu': True, 'test_evaluation_frozen': True, 'CPU_gate_required': {'fits': 135, 'loads': 405, 'version': 12}, 'all360_completion_claimed': False}
        return create(target, value)
    except BaseException as error:
        create(ROOT / ('registration_failure_' + str(time.time_ns()) + '.json'), {'source': S.desc(SOURCE), 'type': type(error).__name__, 'error': str(error), 'sidecars_created': sidecars, 'registered_plan_complete': False, 'captured_at': S.now()})
        raise

def acquire(registration):
    Q.check_pause()
    Q.check_boundary()
    value = S.read(S.checked(registration))
    require(value['schema'] == SCHEMA and value['version'] == 10 and (value['sources'] == sources()), 'Exact new Q10 registration/source set differs')
    require(not (ROOT / 'owner.json').exists(), 'Actual Q10 owner already present; do not duplicate queue')
    inventory = Q.host_inventory()
    roles = O.queue_roles(inventory, os.getpid())
    Q.admit_host(inventory, roles)
    require(Q.linux_inventory() == [], 'Experimental Linux process remains')
    gpu = Q.gpu_empty()
    owner = {'id': uuid.uuid4().hex, 'pid': os.getpid(), 'native_roles': roles, 'source': S.desc(SOURCE), 'registration': registration, 'execution_version': 10, 'started_at': S.now(), 'GPU_before_queue': gpu, 'child_pid': None, 'child_token': None, 'child_job': None, 'child_native_identity': None}
    create(ROOT / 'owner.json', owner)
    return owner

def linux_path(path):
    path = Path(path).resolve()
    require(path.drive and len(path.drive) == 2, 'Windows workspace drive required')
    return '/mnt/' + path.drive[0].lower() + '/' + path.as_posix()[3:]

def child(condition, repair_registration, owner, resume):
    Q.check_pause()
    require(S.read(ROOT / 'owner.json')['id'] == owner['id'], 'Current Q10 owner changed')
    folder = ROOT / 'attempts' / ('f_' + str(time.time_ns()) + '_' + uuid.uuid4().hex[:8])
    token = uuid.uuid4().hex
    job = condition + '-fit'
    admission = {'token': token, 'job': job, 'condition': condition, 'queue_owner_id': owner['id'], 'source': S.desc(O.SOURCE), 'worker_source': S.desc(SOURCE.with_name('repair_owned_worker_v5.py')), 'formal_source': S.desc(SOURCE.with_name('finetune_v8.py')), 'queue_source': S.desc(SOURCE), 'queue_registration': owner['registration'], 'repair_registration': repair_registration, 'created_at': S.now()}
    binding = create(folder / 'worker_admission.json', admission)
    arguments = ['-B', '-u', '-m', O.WORKER_MODULE, '--run', '--condition', condition, '--repair-registration', repair_registration['path'], '--repair-registration-sha256', repair_registration['sha256'], '--worker-admission', binding['path'], '--worker-admission-sha256', binding['sha256'], '--token', token]
    if resume:
        arguments.append('--resume')
    if condition == D.REPAIR_CONDITIONS[1]:
        command = ['wsl.exe', '-d', 'TRI-DEP-Ubuntu-24.04', '--cd', linux_path(S.WORK), '--exec', linux_path(S.OUT / 'assets/eegmamba_linux_env/bin/python')] + arguments
    else:
        command = [str(S.WORK / 'S00/.venv/Scripts/python.exe')] + arguments
    record = {'condition': condition, 'job': job, 'status': 'running', 'started_at': S.now(), 'source': S.desc(SOURCE), 'worker_admission': binding, 'repair_registration': repair_registration, 'command': command, 'inclusive_parent_seconds_primary_cost': False}
    create(folder / 'status.json', record)
    started = time.monotonic()
    with (folder / 'process.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=S.WORK, stdout=log, stderr=subprocess.STDOUT, creationflags=134217728)
        owner.update(child_pid=process.pid, child_token=token, child_job=job, child_native_identity=None, linux_worker_identity=None, child_native_roles=None)
        S.save(ROOT / 'owner.json', owner)
        deadline = time.monotonic() + 10
        while True:
            items = Q.host_inventory()
            native = next((x for x in items if x['pid'] == process.pid), None)
            if native is not None:
                break
            require(process.poll() is None and time.monotonic() < deadline, 'Native child identity could not be bound; do not duplicate GPU')
            time.sleep(0.1)
        require(native['parent'] == owner['pid'] and token in native['command'], 'Actual child launch tree differs')
        owner['child_native_identity'] = native
        S.save(ROOT / 'owner.json', owner)
        record.update(child_pid=process.pid, child_native_identity=native)
        S.save(folder / 'status.json', record)
        if condition == 'EEGMamba-lastblock-finetune':
            linux_identity = L.wait_capture(module=O.WORKER_MODULE, flags={'--token': token, '--condition': condition, '--worker-admission-sha256': binding['sha256'], '--repair-registration-sha256': repair_registration['sha256']}, is_alive=lambda: process.poll() is None)
            items = Q.host_inventory()
            native_tree = N.capture(items, launcher_pid=process.pid, controller_pid=owner['pid'], module=O.WORKER_MODULE, nonce=token)
            require(native_tree[0] == owner['child_native_identity'], 'Native primary changed during Linux handshake')
            Q.admit_host(items, owner['native_roles'] + native_tree)
            owner.update(linux_worker_identity=linux_identity, child_native_roles=native_tree)
            S.save(ROOT / 'owner.json', owner)
            record.update(child_native_roles=native_tree, linux_worker_identity=linux_identity)
            S.save(folder / 'status.json', record)
        code = process.wait()
    record.update(status='complete' if code == 0 else 'failed', returncode=code, finished_at=S.now(), inclusive_parent_seconds=time.monotonic() - started)
    S.save(folder / 'status.json', record)
    owner.update(child_pid=None, child_job=None, child_token=None, child_native_identity=None, linux_worker_identity=None, child_native_roles=None)
    S.save(ROOT / 'owner.json', owner)
    require(code == 0, 'Actual repair worker failed; retain old/source/cost and inspect full checkpoint')
    for seed in (11, 22, 33):
        for fold in (1, 2, 3, 4, 5):
            status = R.fit_status(condition, seed, fold)
            require(status['status'] == 'training_complete', 'Worker returned without actual15 completions')
            R._validate_complete(status['evidence'], R.fit_directory(condition, seed, fold))
    return S.desc(folder / 'status.json')

def cpu_gate():
    for item in S.read(ROOT / 'registration.json')['sources']:
        S.checked(item)
    require(R.status_summary()['statuses'].get('training_complete') == 360, 'All original360 actual fits required')
    path = S.REPORT / 'audit_completed_finetune_checkpoints_v12.py'
    result = subprocess.run([str(S.WORK / 'S00/.venv/Scripts/python.exe'), '-B', '-u', str(path), '--run'], cwd=S.WORK, capture_output=True, text=True, check=True)
    output = json.loads(result.stdout.strip().splitlines()[-1])
    report = S.read(S.checked(output['artifact']))
    require(report['auditor_source'] == S.desc(path) and report['verified_completed_finetune_fits'] == 135 and (report['checkpoint_loads'] == 405) and (report['errors'] == []) and (report['deferred_completed_fits'] == []) and (report['pending_fits'] == []) and (report['all_135_finetune_checkpoints_verified'] is True), 'Actual CPU12 full135/405 gate failed')
    return create(ROOT / ('pre_test_CPU_gate_' + str(time.time_ns()) + '.json'), {'passed': True, 'source': S.desc(SOURCE), 'registration': S.desc(ROOT / 'registration.json'), 'CPU_audit': output['artifact'], 'original_training_complete': 360, 'FT': 135, 'CPU_loads': 405, 'outer_test_probabilities_read': False, 'evaluation_started': False, 'captured_at': S.now()})

def run(*, resume=False):
    binding = S.desc(ROOT / 'registration.json')
    owner = acquire(binding)
    record = {'status': 'running', 'source': S.desc(SOURCE), 'execution_version': 10, 'registration': binding, 'started_at': S.now(), 'outcomes': {}, 'test_evaluation_frozen': True, 'all_plan_complete': False}
    S.save(ROOT / 'status.json', record)
    try:
        value = S.read(S.checked(binding))
        for condition in D.REPAIR_CONDITIONS:
            record['outcomes'][condition] = child(condition, value['repair_registrations'][condition], owner, resume)
            S.save(ROOT / 'status.json', record)
        gate = cpu_gate()
        record.update(status='all_training_CPU_gate_complete_evaluation_pending', CPU_gate=gate, finished_at=S.now(), test_evaluation_frozen=True, all_plan_complete=False)
        S.save(ROOT / 'status.json', record)
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error), finished_at=S.now())
        S.save(ROOT / 'status.json', record)
        raise
    finally:
        current = S.read(ROOT / 'owner.json')
        if current['id'] == owner['id'] and current.get('child_pid') is None:
            create(ROOT / ('released_owner_' + str(time.time_ns()) + '.json'), current)
            (ROOT / 'owner.json').unlink()

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--register', action='store_true')
    for prefix in ('codebrain-report', 'mamba-report', 'codebrain-review', 'mamba-review', 'source-review'):
        parser.add_argument('--' + prefix)
        parser.add_argument('--' + prefix + '-sha256')
    args = parser.parse_args(argv)
    require(not (args.run and args.register), 'Registration and run are separate actual actions')
    if not args.run and (not args.register):
        require(not args.resume, 'Actual resume requires --run')
        print(json.dumps({'status': 'preparation_only', 'model_calls': 0, 'GPU_queries': 0, 'new_fit_or_registration_writes': 0, 'all_plan_complete': False, 'new_qualified_reports_and_complete_source_chain_required': True}))
        return
    if args.register:

        def binding(prefix):
            name = prefix.replace('-', '_')
            path = getattr(args, name)
            digest = getattr(args, name + '_sha256')
            require(path is not None and digest is not None, 'Actual bound evidence required: ' + prefix)
            return {'path': path, 'sha256': digest}
        return register({D.REPAIR_CONDITIONS[0]: binding('codebrain-report'), D.REPAIR_CONDITIONS[1]: binding('mamba-report')}, {D.REPAIR_CONDITIONS[0]: binding('codebrain-review'), D.REPAIR_CONDITIONS[1]: binding('mamba-review')}, binding('source-review'))
    return run(resume=args.resume)
if __name__ == '__main__':
    main()
