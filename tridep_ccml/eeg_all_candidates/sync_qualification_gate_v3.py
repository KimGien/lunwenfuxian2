"""Read-only recovered-Q5 boundary checks for unchanged zero-step probes.

Only control lineage changes from the Q4 gate: actual recovery, dynamic held
parent and completed children. Tensor measurement code stays in V1 delegates.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import types

SOURCE = Path(__file__).resolve()
WORK = SOURCE.parents[2]
OUT = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
REPORT = WORK / 'output/analysis/eeg_all_candidates_v1_20261008'
GUARDIAN = REPORT / 'sync_qualification_boundary_guard_v3.py'
OWNER = OUT / 'execution_acceleration_v1/sync_qualification_owner.json'
CONTROL = OUT / 'execution_acceleration_v1/sync_boundary_v3/registration.json'
DELEGATE = SOURCE.with_name('sync_qualification_gate_v1.py')
DELEGATE_SHA = '0fdb296e7b77e0d1ece1613de66e91884a508cc81126da361cb406edeb8f3b8e'
GPU_OBSERVER = REPORT / 'qualify_acceleration_v2.py'
GPU_OBSERVER_SHA = '3f63b8eb63c33782a972494af0278e8924fa0b866511c7b217f16a2d6e4416e2'
REGISTRY_SHA = '27bf2972c08f77457959cf3cba5bc6056c320f223c2426af5da3ae88d55ce090'


def require(value, message):
    if not value: raise ValueError(message)


def descriptor(path):
    path = Path(path).resolve()
    require(path.is_relative_to(WORK) and path.suffix in ('.json', '.py'),
            'Boundary gate reads only workspace JSON/source')
    return {'path': path.relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read_bound(binding):
    require(type(binding) is dict and set(binding) == {'path', 'sha256'}, 'Exact boundary descriptor required')
    path = (WORK / binding['path']).resolve()
    require(path.suffix == '.json' and descriptor(path) == binding, 'Boundary evidence changed')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def gpu_idle():
    require(descriptor(GPU_OBSERVER)['sha256'] == GPU_OBSERVER_SHA, 'Reviewed all-context GPU observer changed')
    spec = importlib.util.spec_from_file_location('_sync3_readonly_gpu_inventory', GPU_OBSERVER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    observed = module.gpu_observation()
    require(observed.get('idle') is True and observed.get('contexts') == [],
            'Performance qualification requires no GPU context of any type')
    return {**observed, 'gpu_observer_source': descriptor(GPU_OBSERVER)}


def verify_shape(proof, registration, queue_binding, registry_binding, split):
    require(proof.get('schema') == 'sync_qualification_safe_boundary_v3' and proof.get('passed') is True
            and proof.get('queue') == 'continuation_queue_v5' and proof.get('registration') == queue_binding
            and proof.get('original_training_registration') == registry_binding,
            'Actual recovered Q5 natural child boundary identity differs')
    require(all(proof.get(k) is True for k in ('gpu_children_exit_verified', 'held_dispatcher_running_suspended',
            'gpu_idle', 'zero_optimizer_probe_only')), 'Exit0 children, held parent and idle GPU proof required')
    require(type(proof.get('completed_CSBrain_fits_verified')) is int and proof['completed_CSBrain_fits_verified'] == 15,
            'Actual complete CSBrain15 boundary required')
    require((proof.get('condition'), proof.get('seed'), proof.get('fold')) == ('CSBrain-lastblock-finetune', 22, 5)
            and type(proof['seed']) is int and type(proof['fold']) is int
            and proof.get('registered_fold_ids') == split, 'Fixed completed22/fold5 roles differ')
    expected = {f'CSBrain-lastblock-finetune/seed_{seed}/fold_{fold}' for seed in (11,22,33) for fold in range(1,6)}
    rows = [row for row in registration['fits'] if row['condition'] == 'CSBrain-lastblock-finetune']
    actual = {row['fit_id'] for row in rows}
    require(expected == actual and len(rows) == len(actual) == 15, 'Original fifteen unique formal identities differ')
    return expected


def preflight(proof_binding, builder_binding=None):
    proof = read_bound(proof_binding)
    source = descriptor(GUARDIAN)
    require(proof.get('source') == source and (builder_binding is None or builder_binding == source),
            'Actual reviewed guardian3 source differs')
    experimental = proof.get('experimental_sources', [])
    require(len({row['path'] for row in experimental}) == len(experimental), 'Repeated experimental source')
    required = [SOURCE, REPORT/'qualify_sync_reduction_v3.py', REPORT/'qualify_training_sync_v3.py',
                SOURCE.with_name('sync_reduction_v1.py'), SOURCE.with_name('sync_subject_update_v1.py'),
                SOURCE.with_name('sync_state_digest_v1.py')]
    require(all(descriptor(p) in experimental for p in required), 'All real probe/gate/prototype sources must be bound')
    for row in experimental: require(descriptor(WORK/row['path']) == row, 'Experimental source changed')
    from tridep_ccml.eeg_all_candidates import lost_f3_closure_v1 as L
    lineage = L.verify_recovery_registration(proof['recovery_registration'])
    require(len(lineage['value']['sources']) == 80, 'Actual recovery must preserve exactly its eighty sources')
    queue_binding = lineage['base_queue_registration']
    registry_binding = descriptor(OUT/'registration.json')
    require(registry_binding['sha256'] == REGISTRY_SHA, 'Original360 source changed')
    registration = read_bound(registry_binding)
    require(proof['control_registration'] == descriptor(CONTROL), 'Actual optional control registration changed')
    control = read_bound(proof['control_registration'])
    require(control.get('schema') == 'recovered_q5_sync_control_registration_v3'
            and type(control.get('version')) is int and control['version'] == 3
            and control.get('source') == source and control.get('registration') == queue_binding
            and control.get('recovery_registration') == proof['recovery_registration']
            and control.get('recovery_closure') == proof['recovery_closure']
            and control.get('experimental_sources') == experimental,
            'Registered real recovery/control source lineage differs')
    require(control.get('protected_sources') == lineage['value']['sources'],
            'Control must preserve all eighty real recovery sources')
    owner_binding = descriptor(OUT/'continuation_queue_v5/owner.json')
    require(proof['queue_owner'] == owner_binding, 'Held actual Q5 owner changed')
    owner = read_bound(owner_binding)
    require(owner.get('source') == descriptor(WORK/'tridep_ccml/eeg_all_candidates/continuation_queue_v5.py')
            and owner.get('execution_registration') == queue_binding and type(owner.get('execution_version')) is int
            and owner['execution_version'] == 5 and type(owner.get('child_pid')) is int
            and owner.get('child_job') == 'CSBrain-lastblock-finetune-fit'
            and owner.get('recovery_registration') == proof['recovery_registration']
            and owner.get('recovery_execution_source') == lineage['source']
            and owner.get('recovery_closure') == proof['recovery_closure'], 'Actual core5 and recovery owner fields differ')
    closure = read_bound(proof['recovery_closure'])
    closed = L.verify_lost_f3_attempt(closure['original_attempt'], read_bound(closure['original_attempt']),
            {'condition':'CSBrain-lastblock-finetune','seed':33,'fold':4})
    require(type(closed) is dict and closed.get('passed') is True and closed.get('closure') == proof['recovery_closure']
            and closed.get('recovery_registration') == proof['recovery_registration'],
            'Retained process-loss closure must verify without checkpoint reloads')
    expected = verify_shape(proof, registration, queue_binding, registry_binding, registration['folds']['fold_5'])
    statuses = []
    for fit_id in sorted(expected):
        binding = descriptor(OUT/'runs'/fit_id/'status.json'); status = read_bound(binding)
        require(status.get('status') == 'training_complete', 'A registered CSBrain fit is not actually complete')
        statuses.append({'fit_id':fit_id, 'status':binding})
    folder = OUT/'runs/CSBrain-lastblock-finetune/seed_22/fold_5'
    require(proof['checkpoint']['path'] == (folder/'latest.pt').relative_to(WORK).as_posix()
            and proof['train_history']['path'] == (folder/'train_history.json').relative_to(WORK).as_posix(),
            'Only exact completed22/fold5 input evidence is allowed')
    history = read_bound(proof['train_history'])
    require(history and all(type(row['epoch']) is int and row['epoch'] == i for i,row in enumerate(history,1)),
            'Full sequential completed history required')
    spec = importlib.util.spec_from_file_location('_readonly_sync3_live_guardian', GUARDIAN)
    guardian = importlib.util.module_from_spec(spec); spec.loader.exec_module(guardian)
    current = guardian.verify_live_boundary(proof_binding)
    require(current.get('passed') is True and current.get('gpu_children_exit_verified') is True
            and current.get('held_dispatcher_running_suspended') is True, 'Live dynamic guardian boundary failed')
    idle = gpu_idle()
    require(descriptor(OUT/'continuation_queue_v5/owner.json') == owner_binding and descriptor(GUARDIAN) == source,
            'Boundary changed during source-only preflight')
    fit, = [row for row in registration['fits'] if row['fit_id'] == 'CSBrain-lastblock-finetune/seed_22/fold_5']
    return proof, registration, fit, history, {'guardian_live_verification':current, 'gpu_idle_check':idle,
            'completed_condition_statuses':statuses, 'actual_shared_gate_source':descriptor(SOURCE),
            'original_control_reference_source':descriptor(DELEGATE), 'actual_recovery_registration':proof['recovery_registration'],
            'same_original360_roles_and_probe_budgets':True, 'checkpoint_tensor_loads_this_gate':0}
