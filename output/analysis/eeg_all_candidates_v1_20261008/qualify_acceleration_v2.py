"""Q5 optional batching measurement, with explicit zero-model GPU deferral.

The pinned V1 run body retains every tensor, diagnostic, role and numerical
comparison operation. Only the queue gate, record provenance and deferred
pre-GPU outcome differ. Results never authorize expanded chunks in F3/B1.
No --run: stdlib-only pending evidence, without process/GPU/data queries.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
import types
import xml.etree.ElementTree as ET

SOURCE = Path(__file__).resolve()
REPORT = SOURCE.parent
WORK = REPORT.parents[2]
MOD = WORK / 'tridep_ccml/eeg_all_candidates'
OUT = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
ROOT = OUT / 'execution_acceleration_v1'
DELEGATE = REPORT / 'qualify_acceleration_20261009.py'
DELEGATE_SHA = '878fda0f97d0a6ec801501baf1bad084d77d2d319897609a2a39600e01568a5e'
BUILDER = MOD / 'acceleration_prepare_v2.py'
QUEUE = 'continuation_queue_v5'
Q4_SHA = 'c831bca3afad5574a7bc084fa4a0ebcaec7bea629679d7c1b426171092f8b089'
OWNER = ROOT / 'qualification_owner.json'


def load_delegate():
    import hashlib
    data = DELEGATE.read_bytes()
    if hashlib.sha256(data).hexdigest() != DELEGATE_SHA:
        raise ValueError('Sealed V1 measurement implementation changed')
    module = types.ModuleType('isolated_q5_batching_measurement_v1')
    # Runtime provenance is V2; the unchanged compiled body is explicitly bound
    # separately in every output. The imported sealed module is never patched.
    module.__file__ = str(SOURCE)
    exec(compile(data, str(DELEGATE), 'exec'), module.__dict__)
    return module


D = load_delegate()
require = D.require
descriptor, checked, json_read = D.descriptor, D.checked, D.json_read
atomic_json = D.atomic_json


class ExternalGPUContext(RuntimeError):
    def __init__(self, observation):
        super().__init__('Optional batching measurement deferred: external GPU context observed')
        self.observation = observation


def parse_gpu_xml(stdout):
    """All reported GPU process types count, including graphics and C+G."""
    root = ET.fromstring(stdout)
    require(root.tag == 'nvidia_smi_log', 'Unexpected NVIDIA process report')
    devices = root.findall('gpu')
    require(devices, 'NVIDIA report contains no GPU')
    contexts = []
    for device in devices:
        processes = device.find('processes')
        require(processes is not None, 'NVIDIA process table unavailable')
        # N/A is not an observed empty process table.
        require((processes.text or '').strip() not in ('N/A', '[Not Supported]'), 'NVIDIA process table unsupported')
        for process in processes.findall('process_info'):
            pid_text = process.findtext('pid', '').strip()
            require(pid_text.isdecimal() and int(pid_text) > 0, 'Unresolved GPU process identity')
            contexts.append({'gpu': device.get('id'), 'pid': int(pid_text),
                             'type': process.findtext('type'), 'process_name': process.findtext('process_name'),
                             'used_memory': process.findtext('used_memory')})
    return contexts


def gpu_observation():
    query = subprocess.run(['nvidia-smi', '-q', '-x'], capture_output=True, text=True, timeout=20, check=True)
    contexts = parse_gpu_xml(query.stdout)
    return {'query': 'nvidia-smi -q -x', 'checked_at': datetime.now(timezone.utc).isoformat(),
            'stdout': query.stdout, 'stderr': query.stderr, 'contexts': contexts, 'idle': not contexts,
            'scope': 'all reported GPU process types, including graphics; no processes modified'}


def strict_idle():
    observed = gpu_observation()
    if not observed['idle']:
        raise ExternalGPUContext(observed)
    return observed


def queue_context(condition, *, linux_owner=None):
    """Source/identity checks precede optional-device discovery; no tensors."""
    require(type(condition) is str and '/' not in condition and '\\' not in condition and '..' not in condition,
            'Invalid condition identity')
    queue_path = OUT / QUEUE / 'registration.json'
    queue_binding = descriptor(queue_path)
    queue = json_read(queue_path)
    require(type(queue.get('version')) is int and queue['version'] == 5, 'Actual Q5 registration required')
    require(queue.get('registry') == descriptor(D.REGISTRY), 'Q5 original registration differs')
    sources = queue.get('sources')
    require(type(sources) is list and len({b['path'] for b in sources}) == len(sources), 'Q5 source list missing/duplicated')
    require(descriptor(SOURCE) in sources and descriptor(BUILDER) in sources, 'Q5 must bind both actual preparation/qualification sources')
    require(descriptor(MOD / (QUEUE + '.py')) in sources, 'Actual Q5 controller source missing')
    for binding in sources:
        checked(binding)
    q4_binding = descriptor(OUT / 'continuation_queue_v4/registration.json')
    require(q4_binding['sha256'] == Q4_SHA and queue.get('previous_execution_registration') == q4_binding,
            'Q5 must preserve the sealed actual Q4 registration')
    previous = json_read(checked(q4_binding))
    require(type(previous.get('sources')) is list and len(previous['sources']) == 59
            and all(binding in sources for binding in previous['sources']), 'Q5 must preserve all59 Q4 sources')
    require(descriptor(D.REGISTRY)['sha256'] == D.REGISTRY_SHA, 'Original360 registration changed')
    for name, digest in D.DELEGATE_HASHES.items():
        require(descriptor(MOD / name)['sha256'] == digest, 'Sealed mathematical delegate changed')
    registry = json_read(D.REGISTRY)
    fits = [f for f in registry['fits'] if f['condition'] == condition and f['stage'] == '3' and f['seed'] == 11 and f['fold'] == 1]
    require(len(fits) == 1, 'Registered FT seed11/fold1 identity required')
    owner_path = OUT / QUEUE / 'owner.json'
    owner_binding = descriptor(owner_path)
    owner = json_read(owner_path)
    require(owner.get('source') == descriptor(MOD / (QUEUE + '.py'))
            and owner.get('execution_registration') == queue_binding
            and type(owner.get('execution_version')) is int and owner['execution_version'] == 5,
            'Queue owner must bind actual Q5 source and registration')
    expected_job = condition + '-acceleration-prepare'
    require(owner.get('child_job') == expected_job, 'Q5 job is not this condition preparation')
    windows_live = D.pid_alive if os.name == 'nt' else lambda pid: D.windows_identity(pid) is not None
    require(windows_live(owner['pid']), 'Q5 dispatcher is not live')
    if os.name == 'nt':
        require(owner.get('child_pid') in (os.getpid(), os.getppid()), 'Qualifier is not the actual Q5 child/launcher')
    else:
        require(linux_owner is not None, 'Linux qualification requires its bound actual owner')
        linux = json_read(checked(linux_owner))
        require(linux.get('pid') in (os.getpid(), os.getppid()) and D.pid_alive(linux['pid']), 'Wrong actual Linux qualifier/launcher')
        require(type(owner.get('child_pid')) is int, 'Windows WSL launcher missing')
    parent_identity = D.windows_identity(owner['pid'])
    child_identity = D.windows_identity(owner['child_pid'])
    require(parent_identity is not None and child_identity is not None, 'Q5 CIM parent/child identity unavailable')
    require(child_identity.get('ParentProcessId') == owner['pid'], 'Q5 child ancestry differs')
    for folder in ('continuation_queue', 'continuation_queue_v2', 'continuation_queue_v3', 'continuation_queue_v4'):
        path = OUT / folder / 'owner.json'
        if path.exists():
            old = json_read(path)
            for key in ('pid', 'child_pid'):
                if old.get(key) is not None:
                    require(not windows_live(old[key]), 'Another registered training queue is still live')
    context = {'queue': QUEUE, 'queue_registration': queue_binding, 'queue_owner': owner_binding,
               'queue_child_job': expected_job, 'queue_parent_identity': parent_identity,
               'queue_child_identity': child_identity}
    if linux_owner is not None:
        context.update(linux_owner=linux_owner, windows_queue_owner=owner_binding)
    # Recheck after identity observations; never let a different job inherit it.
    checked(owner_binding)
    return context, registry, fits[0]


def preflight(proof_path):
    proof = json_read(proof_path)
    require(proof.get('schema') == 'acceleration_safe_stop_v2'
            and proof.get('mode') == 'queue_exclusive_post_prepare', 'Only actual Q5 fresh preparation proof is supported')
    require(checked(proof['source']) == BUILDER.resolve(), 'Actual AP2 proof builder required')
    require(proof.get('registration') == descriptor(D.REGISTRY), 'Proof original registration differs')
    require(type(proof.get('seed')) is int and proof['seed'] == 11
            and type(proof.get('fold')) is int and proof['fold'] == 1, 'Fresh qualification identity differs')
    require(proof.get('checkpoints') == [] and proof.get('stopped_process_ids') == []
            and proof.get('zero_optimizer_qualification_only') is True, 'Fresh zero-update probe only')
    context, registry, fit = queue_context(proof['condition'], linux_owner=proof.get('linux_owner'))
    for key, value in context.items():
        require(proof.get(key) == value, 'Queue proof identity changed: ' + key)
    observed = strict_idle()
    # A busy-at-creation proof cannot become an authorization by a later race.
    if proof.get('gpu_idle') is False:
        prior = proof.get('gpu_observation', {})
        require(prior.get('idle') is False and prior.get('contexts'), 'Missing actual original deferral observation')
        raise ExternalGPUContext(prior)
    require(proof.get('passed') is True and proof.get('gpu_idle') is True, 'Explicit idle proof missing')
    return {**proof, '_actual_queue_child_identity': context['queue_child_identity']}, registry, fit, registry['folds']['fold_1'], [], None, observed


def provenance(value):
    value = dict(value)
    if value.get('schema') in ('diagnostic_batching_qualification_attempt_v1', 'diagnostic_batching_qualification_v1'):
        value['schema'] = value['schema'][:-2] + 'v2'
        value['qualifier_source'] = descriptor(SOURCE)
        value['delegated_measurement_source'] = descriptor(DELEGATE)
        value['actual_queue'] = QUEUE
        value['expanded_batch_activation_authorized'] = False
        value['production_diagnostic_chunk'] = 8
        value['formal_optimizer_updates'] = 0
        value['nested_measurement_timings_are_additive'] = False
        peaks = [m['peak_allocated_bytes'] for m in value.get('measurements', [])
                 if type(m.get('peak_allocated_bytes')) is int and m['peak_allocated_bytes'] >= 0]
        failure_peak = value.get('peak_allocated_bytes_observed_at_failure')
        if type(failure_peak) is int and failure_peak >= 0:
            peaks.append(failure_peak)
        value['peak_allocated_bytes_observed'] = max(peaks) if peaks else None
        value['peak_scope'] = ('no_GPU_or_model_work' if value.get('status') == 'deferred' else
            'maximum recorded repetition/failure allocation peak; construction-only transient peak is not measured')
        if value.get('schema') == 'diagnostic_batching_qualification_attempt_v2':
            value.setdefault('model_constructions', 1 if value.get('status') == 'complete' else None)
            value['checkpoint_weight_deserializations'] = 0  # Q5 fresh-initialization branch only.
        if 'source_bindings' in value:
            value['source_bindings'] = {**value['source_bindings'], 'qualifier_source': descriptor(SOURCE),
                                      'delegated_measurement_source': descriptor(DELEGATE)}
    return value


def write_actual(path, value, exclusive=False):
    atomic_json(path, provenance(value), exclusive=exclusive)


def execute_explicit(args):
    require(args.stop_proof is not None, 'Explicit Q5 proof required')
    require(type(args.repeats) is int and 2 <= args.repeats <= 5, 'Use2..5 measured repeats after one warmup')
    proof = json_read(args.stop_proof)
    condition = proof['condition']
    require(type(condition) is str and '/' not in condition and '\\' not in condition and '..' not in condition, 'Invalid condition path')
    attempt_path = ROOT / condition / 'qualification_attempts_v2' / ('attempt_' + str(time.time_ns())) / 'attempt.json'
    started = time.perf_counter()
    trace = {'schema': 'diagnostic_batching_qualification_attempt_v1', 'kind': 'engineering_qualification', 'status': 'running',
             'started_at': datetime.now(timezone.utc).isoformat(), '_timer_start': started,
             'qualifier_source': descriptor(SOURCE), 'thresholds': D.THRESHOLDS,
             'safe_stop_proof': descriptor(args.stop_proof), 'condition': condition, 'seed': proof.get('seed'), 'fold': proof.get('fold'),
             'forward_started': 0, 'forward_returned': 0, 'encoder_forward_started': 0, 'encoder_forward_returned': 0,
             'encoder_windows_started': 0, 'encoder_windows_returned': 0, 'backward_calls': 0, 'optimizer_updates': 0,
             'forbidden_backward_attempts': 0, 'forbidden_optimizer_attempts': 0, 'test_forward_count': 0,
             'counter_durability': 'Phase/repetition snapshots; hard-crash unflushed tail is unknown, never zero-filled',
             'unflushed_crash_tail_exact': None, 'gpu_stage_entered': False,
             'peak_allocated_bytes_observed_at_failure': None}
    write_actual(attempt_path, trace, exclusive=True)
    try:
        # Only this isolated delegate is adjusted. Its tensor run body is the
        # pinned V1 implementation, including its second idle check after lease.
        D.preflight, D.gpu_idle, D.atomic_json = preflight, strict_idle, write_actual
        result = D.run(args, attempt_path, trace)
        return {**result, 'status': 'complete', 'expanded_batch_activation_authorized': False}
    except ExternalGPUContext as error:
        require(trace['gpu_stage_entered'] is False and trace['forward_started'] == 0,
                'External-context deferral cannot erase actual model work')
        trace.pop('_timer_start', None)
        trace.update(status='deferred', phase='optional_probe_deferred_before_gpu', finished_at=datetime.now(timezone.utc).isoformat(),
                     seconds=time.perf_counter() - started, external_gpu_observation=error.observation,
                     reason='external_gpu_context_before_optional_measurement', qualification_performed=False,
                     model_constructions=0, raw_reads=0, checkpoint_reads=0, torch_imports=0,
                     production_diagnostic_chunk=8, mandatory_training_blocked_by_this_deferral=False,
                     unflushed_crash_tail_exact=0)
        write_actual(attempt_path, trace)
        return {'artifact': descriptor(attempt_path), 'status': 'deferred', 'passed': False, 'selected_chunk': None,
                'expanded_batch_activation_authorized': False}
    except BaseException as error:
        timer = trace.pop('_timer_start', None)
        trace.update(status='failed', error_type=type(error).__name__, error=str(error), finished_at=datetime.now(timezone.utc).isoformat())
        if timer is not None:
            trace['seconds'] = time.perf_counter() - timer
        write_actual(attempt_path, trace)
        raise


def run_qualification(condition, seed, fold, checkpoint=None, *, safe_stop_proof, repeats=2):
    require(checkpoint is None, 'Q5 optional preparation only measures fresh initialization')
    proof_path = checked(safe_stop_proof)
    proof = json_read(proof_path)
    require(type(seed) is int and type(fold) is int and (condition, seed, fold) == (proof['condition'], proof['seed'], proof['fold']), 'API/proof identity differs')
    return execute_explicit(types.SimpleNamespace(stop_proof=proof_path, repeats=repeats))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--stop-proof', type=Path)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args(argv)
    if not args.run:
        value = {'schema': 'diagnostic_batching_qualification_pending_v2', 'status': 'pending_explicit_Q5_run',
                 'qualifier_source': descriptor(SOURCE), 'delegated_measurement_source': descriptor(DELEGATE),
                 'torch_imports': 0, 'model_constructions': 0, 'checkpoint_reads': 0, 'raw_reads': 0, 'GPU': 0,
                 'optimizer_updates': 0, 'production_diagnostic_chunk': 8, 'expanded_batch_activation_authorized': False}
        path = REPORT / ('diagnostic_batching_qualification_v2_pending_' + str(time.time_ns()) + '.json')
        atomic_json(path, value, exclusive=True)
        print(json.dumps({'artifact': descriptor(path), 'status': value['status']})); return 0
    print(json.dumps(execute_explicit(args))); return 0


if __name__ == '__main__':
    raise SystemExit(main())
