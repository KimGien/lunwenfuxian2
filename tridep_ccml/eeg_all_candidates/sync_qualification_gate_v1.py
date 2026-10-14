"""Read-only shared natural-boundary gate; no tensor imports or process control."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

SOURCE = Path(__file__).resolve()
WORK = SOURCE.parents[2]
OUT = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
REPORT = WORK / 'output/analysis/eeg_all_candidates_v1_20261008'
GUARDIAN = REPORT / 'sync_qualification_boundary_guard_v1.py'
OWNER = OUT / 'execution_acceleration_v1/sync_qualification_owner.json'
REGISTRY_SHA = '27bf2972c08f77457959cf3cba5bc6056c320f223c2426af5da3ae88d55ce090'


def require(value, message):
    if not value: raise ValueError(message)


def descriptor(path):
    path = Path(path).resolve()
    require(path.is_relative_to(WORK) and path.suffix in ('.json', '.py'), 'Boundary gate reads only workspace JSON/source')
    return {'path': path.relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read_bound(binding):
    require(type(binding) is dict and set(binding) == {'path', 'sha256'}, 'Exact boundary descriptor required')
    path = (WORK / binding['path']).resolve()
    require(descriptor(path) == binding and path.suffix == '.json', 'Boundary evidence bytes changed')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def gpu_idle():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=20)
    require(not result.stdout.strip(), 'Another GPU compute process is present')
    return {'query': 'nvidia-smi compute-apps', 'stdout': result.stdout, 'stderr': result.stderr}


def verify_shape(proof, registration, q4_binding, registry_binding, split):
    require(proof.get('schema') == 'sync_qualification_safe_boundary_v1' and proof.get('passed') is True
            and proof.get('queue') == 'continuation_queue_v4' and proof.get('registration') == q4_binding
            and proof.get('original_training_registration') == registry_binding, 'Original/Q4 natural boundary identity differs')
    require(all(proof.get(k) is True for k in ('gpu_children_exit_verified', 'held_dispatcher_running_suspended',
            'gpu_idle', 'zero_optimizer_probe_only')), 'Completed GPU exit and suspended dispatcher proof required')
    require(type(proof.get('completed_CSBrain_fits_verified')) is int and proof['completed_CSBrain_fits_verified'] == 15,
            'Natural full CSBrain15 boundary required')
    require((proof.get('condition'), proof.get('seed'), proof.get('fold')) == ('CSBrain-lastblock-finetune', 22, 5)
            and type(proof['seed']) is int and type(proof['fold']) is int
            and proof.get('registered_fold_ids') == split, 'Fixed completed seed22/fold5 identity or roles differ')
    expected = {f"CSBrain-lastblock-finetune/seed_{seed}/fold_{fold}" for seed in (11, 22, 33) for fold in range(1, 6)}
    actual = {r['fit_id'] for r in registration['fits'] if r['condition'] == 'CSBrain-lastblock-finetune'}
    require(actual == expected, 'Original CSBrain15 registration identity differs')
    return expected


def preflight(proof_binding, builder_binding=None):
    """Returns proof, original registration, fixed fit, history, current guard evidence.

    The guardian retains the actual process handles throughout its sequential
    child calls. Its exported verifier performs only read-only identity checks.
    """
    proof = read_bound(proof_binding)
    builder_binding = proof.get('source') if builder_binding is None else builder_binding
    require(builder_binding == proof['source'] and descriptor(GUARDIAN) == builder_binding, 'Reviewed guardian source differs')
    experimental = proof.get('experimental_sources', [])
    require(descriptor(SOURCE) in experimental and len({row['path'] for row in experimental}) == len(experimental),
            'Guardian must bind this shared gate and unique experimental sources')
    for binding in experimental:
        require(descriptor(WORK / binding['path']) == binding, 'An inspected experimental source changed')
    registry_binding = descriptor(OUT / 'registration.json')
    require(registry_binding['sha256'] == REGISTRY_SHA, 'Original360 registration differs')
    registration = read_bound(registry_binding)
    q4_binding = descriptor(OUT / 'continuation_queue_v4/registration.json')
    q4 = read_bound(q4_binding)
    require(q4['registry'] == registry_binding and q4['version'] == 4, 'Q4 registration lineage differs')
    for binding in q4['sources']:
        require(descriptor(WORK / binding['path']) == binding, 'A sealed Q4 source changed')
    split = registration['folds']['fold_5']
    expected = verify_shape(proof, registration, q4_binding, registry_binding, split)
    statuses = []
    for fit_id in sorted(expected):
        path = OUT / 'runs' / fit_id / 'status.json'
        binding = descriptor(path); value = read_bound(binding)
        require(value['status'] == 'training_complete', 'A CSBrain fit is not actually complete')
        statuses.append({'fit_id': fit_id, 'status': binding})
    owner_binding = descriptor(OUT / 'continuation_queue_v4/owner.json')
    require(owner_binding == proof['queue_owner'], 'Held Q4 owner bytes changed')
    owner = read_bound(owner_binding)
    require(owner.get('child_pid') is not None, 'Boundary must bind the just-exited child identity')
    folder = OUT / 'runs/CSBrain-lastblock-finetune/seed_22/fold_5'
    require(proof['checkpoint']['path'] == (folder / 'latest.pt').relative_to(WORK).as_posix()
            and proof['train_history']['path'] == (folder / 'train_history.json').relative_to(WORK).as_posix(), 'Only completed original seed22fold5 evidence allowed')
    history = read_bound(proof['train_history'])
    require(history and all(row['epoch'] == i for i, row in enumerate(history, 1)), 'Complete sequential train history required')
    spec = importlib.util.spec_from_file_location('_read_only_sync_boundary_guard', GUARDIAN)
    guardian = importlib.util.module_from_spec(spec); spec.loader.exec_module(guardian)
    require(callable(getattr(guardian, 'verify_live_boundary', None)), 'Guardian must export read-only verify_live_boundary')
    current = guardian.verify_live_boundary(proof_binding)
    require(type(current) is dict and current.get('passed') is True
            and current.get('gpu_children_exit_verified') is True
            and current.get('held_dispatcher_running_suspended') is True, 'Live guardian identity/suspension verification failed')
    idle = gpu_idle()
    require(descriptor(OUT / 'continuation_queue_v4/owner.json') == owner_binding
            and descriptor(GUARDIAN) == builder_binding, 'Boundary changed during preflight')
    fit, = [row for row in registration['fits'] if row['fit_id'] == 'CSBrain-lastblock-finetune/seed_22/fold_5']
    return proof, registration, fit, history, {'guardian_live_verification': current, 'gpu_idle_check': idle,
                                           'completed_condition_statuses': statuses}
