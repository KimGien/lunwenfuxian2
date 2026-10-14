"""Independent, zero-update head-resource measurements after full queue completion.

Default/check-only mode is safe while training is pending. No torch, feature
array, model, probability or checkpoint import/read occurs on that branch.
Historical engineering artifacts are never modified or backfilled.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

WORK = Path(__file__).resolve().parents[3]
OUT_REL = 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
REPORT_REL = 'output/analysis/eeg_all_candidates_v1_20261008'
SOURCE = Path(__file__).resolve()


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def descriptor(path, work=WORK):
    path, work = Path(path).resolve(), Path(work).resolve()
    require(path.is_relative_to(work), 'Evidence outside workspace')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': path.relative_to(work).as_posix(), 'sha256': digest}


def checked(record, work=WORK):
    require(isinstance(record, dict) and set(record) == {'path', 'sha256'}, 'Invalid evidence descriptor')
    path = (Path(work) / record['path']).resolve()
    require(descriptor(path, work) == record, 'Evidence changed: ' + record['path'])
    return path


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def replace_own_trace(path, value):
    """Replace only this invocation's trace; serialize once before retrying I/O."""
    path = Path(path)
    require(path.is_file(), 'Attempt trace must be created before updates')
    temporary = path.parent / ('.m_' + uuid.uuid4().hex[:12] + '.tmp')
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with temporary.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    for attempt in range(30):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(.1)


def screening(registration, statuses, queue, owner_exists):
    """Pure metadata gate; deliberately cannot open probabilities or import torch."""
    fits = registration.get('fits', [])
    require(registration.get('formal_fits') == 360 and len(fits) == 360
            and len({row['fit_id'] for row in fits}) == 360, 'Full360 identities required')
    require(len(statuses) == 360, 'A status observation is required for every fit')
    counts = dict(Counter(statuses))
    reasons = []
    if counts.get('training_complete') != 360:
        reasons.append('all360 actual training completions pending')
    if queue.get('status') != 'all_training_complete' or queue.get('test_evaluation_completed') is not True:
        reasons.append('Q3 final training/evaluation terminal status pending')
    if queue.get('test_evaluation_frozen') is not False:
        reasons.append('Q3 evaluation gate remains frozen or unknown')
    if owner_exists:
        reasons.append('Q3 owner has not been released')
    return {'status': 'pending' if reasons else 'metadata_ready', 'reasons': reasons,
            'training_status_counts': counts, 'test_or_feature_reads_performed': False}


def existing_test_gate(out, training, *, work=WORK):
    stable = {key: training[key] for key in ('passed', 'registration', 'formal_fits_verified', 'statuses')}
    digest = hashlib.sha256(json.dumps(stable, sort_keys=True, ensure_ascii=False,
                                      separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    path = Path(out) / 'evaluation' / ('test_gate_' + digest + '.json')
    require(path.is_file() and read(path) == stable, 'Existing exact source-bound E2 test gate required; no old evaluation files may be created')
    complete = read(Path(out) / 'evaluation/complete.json')
    binding = descriptor(path, work)
    require(complete.get('test_gate') == binding, 'E2 completion must bind this existing test gate')
    return binding


def inspect_gate(*, work=WORK, verifier=None):
    work = Path(work).resolve(); out = work / OUT_REL
    registration = read(out / 'registration.json')
    statuses = []
    for row in registration['fits']:
        path = out / 'runs' / row['condition'] / ('seed_' + str(row['seed'])) / ('fold_' + str(row['fold'])) / 'status.json'
        statuses.append(read(path)['status'] if path.exists() else 'engineering_pending')
    queue_path = out / 'continuation_queue_v3/status.json'
    queue = read(queue_path) if queue_path.exists() else {}
    queue_binding = descriptor(queue_path, work) if queue_path.exists() else None
    value = screening(registration, statuses, queue, (out / 'continuation_queue_v3/owner.json').exists())
    value.update(checked_at=now(), registration=descriptor(out / 'registration.json', work),
                 model_constructions=0, gpu_calls=0)
    if value['status'] == 'pending':
        return value
    q3_path = out / 'continuation_queue_v3/registration.json'
    q3 = read(q3_path)
    require(len(q3['sources']) == 47 and q3['registry'] == value['registration'], 'Q3 source/registry binding differs')
    for record in q3['sources']:
        checked(record, work)
    expected_jobs = {row['id'] for row in q3['jobs']}
    require(set(queue.get('outcomes', {})) == expected_jobs
            and all(v == 'complete' for v in queue['outcomes'].values()), 'Not every fixed Q3 job completed successfully')
    if verifier is None:
        require(work == WORK, 'Production verifier may only use this actual workspace')
        sys.path.insert(0, str(WORK))
        from tridep_ccml.eeg_all_candidates import registry as R, evaluate_v2 as E
        training = R.assert_test_unlocked()
        require(training['formal_fits_verified'] == 360, 'Full360 byte-bound gate failed')
        gate_binding = existing_test_gate(out, training)
        proof = E.completed_report_verified()
        checked(gate_binding)
    else:
        proof = verifier()
    require(proof.get('passed') is True and proof.get('formal_fits_verified') == 360
            and proof.get('supervised_fits_evaluated') == 345 and proof.get('supervised_conditions') == 23
            and proof.get('model_constructions') == proof.get('gpu_forward_calls') == 0,
            'Read-only E2 full report verification failed')
    require(not (out / 'continuation_queue_v3/owner.json').exists(), 'Q3 owner appeared during verification')
    require(descriptor(queue_path, work) == queue_binding, 'Queue terminal status changed during verification')
    value.update(status='ready_after_full_e2_verification', evaluation_verification=proof,
                 q3_registration=descriptor(q3_path, work), q3_terminal_status=descriptor(queue_path, work),
                 test_or_feature_reads_performed=True,
                 read_scope='Only E2 existing evaluation artifacts after full360 gate; no feature arrays/models/GPU')
    return value


def choose_records(manifest, people, *, key_override=None):
    """Select descriptors before any feature file is opened; never select all37."""
    require(people and len(people) == len(set(people)), 'Probe people must be unique and nonempty')
    records = manifest['records']
    require(len({r['subject_id'] for r in records}) == len(records), 'Duplicate feature person')
    by_id = {row['subject_id']: row for row in records}
    require(set(people).issubset(by_id), 'Probe feature missing')
    return [{'subject_id': sid, 'features': by_id[sid]['features'],
             'archive_key': key_override if key_override is not None else by_id[sid].get('archive_key')}
            for sid in people]


def probe_plan(condition, registration, *, work=WORK):
    work = Path(work).resolve(); out = work / OUT_REL
    row = next(r for r in registration['fits'] if r['condition'] == condition and r['seed'] == 11 and r['fold'] == 1)
    policy = next(c['policy_key'] for c in registration['configuration']['conditions'] if c['condition_id'] == condition)
    require(policy in ('diagnostic', 'frozen'), 'Only diagnostic/frozen head measurements are registered')
    paired = policy == 'diagnostic'; n = 16 if paired else 8
    people = row['train_ids'][:n]
    require(not set(people) & set(row['val_ids'] + row['test_ids']), 'Probe includes non-training people')
    lock_path = out / 'engineering' / (condition + '.json'); lock = read(lock_path); contract = lock['contract']
    require(lock['condition'] == condition and lock['registration'] == descriptor(out / 'registration.json', work), 'Engineering identity differs')
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(lock['configuration_sha256'] == digest, 'Historical engineering contract changed')
    require(contract['probe_subject_ids'] == people, 'Must use the same historical training probe people')
    for source in contract['implementation_sources'] + contract['weight_sources']:
        checked(source, work)
    feature_desc = contract['input_contract']['frozen_cache'] if paired else descriptor(out / 'frozen_features' / condition / 'manifest.json', work)
    require(feature_desc in contract['weight_sources'], 'Feature manifest differs from source-bound engineering contract')
    manifest = read(checked(feature_desc, work))
    if not paired:
        require(manifest['passed'] is True and manifest['condition'] == condition and manifest['encoder_fit_subject_ids'] == [], 'External frozen feature manifest differs')
    key = ('eegpt' if condition.startswith('EEGPT-') else 'labram') if paired else None
    rows = choose_records(manifest, people, key_override=key)
    return {'condition': condition, 'paired': paired, 'dimension': 200 if paired else manifest['dimension'],
            'probe_subject_ids': people, 'feature_manifest': feature_desc, 'feature_records': rows,
            'condition_source_lock': descriptor(lock_path, work), 'contract': contract,
            'labels': [registration['labels'][sid] for sid in people]}


@contextmanager
def exclusive_measurement_lease(out):
    path = Path(out) / 'supervised_engineering/supplemental_head_resource_owner.json'
    value = {'id': uuid.uuid4().hex, 'pid': os.getpid(), 'started_at': now(),
             'subtype': 'supplemental_head_resource_measurement'}
    write_new(path, value)
    try:
        yield value
    finally:
        require(read(path)['id'] == value['id'], 'Supplemental ownership changed; retaining lease')
        path.unlink()


def assert_terminal_still_bound(gate, out):
    require(not (Path(out) / 'continuation_queue_v3/owner.json').exists(), 'A Q3 owner appeared; no supplementary model operation allowed')
    checked(gate['registration']); checked(gate['q3_registration']); checked(gate['q3_terminal_status'])
    checked(gate['evaluation_verification']['complete_manifest'])


def successful_existing(path, *, plan, gate, source):
    record = read(path)
    return (record.get('subtype') == 'supplemental_head_resource_measurement'
            and record.get('condition') == plan['condition'] and record.get('source') == source
            and record.get('registration') == gate['registration'] and record.get('condition_source_lock') == plan['condition_source_lock']
            and type(record.get('seed')) is int and record['seed'] == 11
            and type(record.get('fold')) is int and record['fold'] == 1 and record.get('input_role') == 'train'
            and record.get('probe_subject_ids') == plan['probe_subject_ids']
            and record.get('feature_manifest') == plan['feature_manifest']
            and record.get('loaded_feature_sources') == [r['features'] for r in plan['feature_records']]
            and record.get('feature_archive_keys_read') == [{'subject_id': r['subject_id'], 'key': r['archive_key']} for r in plan['feature_records']]
            and record.get('actual_trainable_parameter_names') == plan['contract']['trainable_parameter_names']
            and record.get('gate', {}).get('evaluation_verification', {}).get('complete_manifest') == gate['evaluation_verification']['complete_manifest']
            and record.get('status') == 'complete' and record.get('passed') is True
            and type(record.get('optimizer_updates')) is int and record['optimizer_updates'] == 0
            and type(record.get('gradient_passes')) is int and record['gradient_passes'] == 1
            and record.get('optimizer_step_started') == record.get('optimizer_step_returned') == 0
            and record.get('formal_fits') == record.get('new_encoder_forwards') == 0
            and record.get('forward_started') == record.get('forward_returned') == 1
            and record.get('backward_started') == record.get('backward_returned') == 1
            and record.get('new_measurement_not_historical_backfill') is True
            and record.get('resume_checked') is True
            and isinstance(record.get('seconds'), (int, float)) and record['seconds'] > 0
            and isinstance(record.get('forward_backward_seconds'), (int, float)) and record['forward_backward_seconds'] > 0
            and isinstance(record.get('peak_allocated_bytes'), int) and not isinstance(record['peak_allocated_bytes'], bool)
            and record['peak_allocated_bytes'] > 0)


def measure(plan, gate):
    """Only called after the full gate, inside the supplementary exclusive lease."""
    out = WORK / OUT_REL; condition = plan['condition']; token = str(time.time_ns()) + '_' + uuid.uuid4().hex[:4]
    folder = out / 'supervised_engineering' / condition
    path = folder / ('measurement_' + token + '.json')
    probe_path = folder / ('probe_' + token + '.pt')
    record = {'subtype': 'supplemental_head_resource_measurement', 'condition': condition, 'seed': 11, 'fold': 1,
              'source': descriptor(SOURCE), 'registration': gate['registration'], 'condition_source_lock': plan['condition_source_lock'],
              'gate': gate, 'probe_subject_ids': plan['probe_subject_ids'], 'input_role': 'train',
              'feature_manifest': plan['feature_manifest'], 'loaded_feature_sources': [], 'feature_archive_keys_read': [],
              'started_at': now(), 'status': 'running', 'passed': False,
              'new_measurement_not_historical_backfill': True, 'historical_resource_fields_modified': False,
              'optimizer_updates': 0, 'gradient_passes': 0, 'formal_fits': 0, 'new_encoder_forwards': 0,
              'forward_started': 0, 'forward_returned': 0, 'backward_started': 0, 'backward_returned': 0,
              'optimizer_step_started': 0, 'optimizer_step_returned': 0,
              'model_constructions_started': 0, 'model_constructions_returned': 0,
              'seconds': None, 'forward_backward_seconds': None, 'peak_allocated_bytes': None,
              'measurement_scope': {'seconds': 'Monotonic attempt wall time including imports, train-only feature IO, CUDA setup, head, one forward/backward, checkpoint serialization/restore and durable trace writes; excludes E2 prerequisite verification.',
                                    'forward_backward_seconds': 'Synchronized monotonic elapsed spanning exactly one forward/loss/backward and their durable started/returned trace writes; not pure GPU-kernel time.',
                                    'peak_allocated_bytes': 'Absolute CUDA allocated-byte high-water mark after reset before seed/model/data allocations, through roundtrip; baseline and reserved-byte peaks also recorded.',
                                    'features': 'Only listed fold1 training people, all60 cached windows; lazy NPZ reads only the specified representation key.',
                                    'no_historical_backfill': 'These are additional measurements and costs; historical missing durations/peaks remain unknown.'}}
    write_new(path, record)
    began = time.perf_counter(); torch = model = optimizer = features = target = logits = loss = saved = restored = None
    old_steps = []; memory_started = False
    try:
        assert_terminal_still_bound(gate, out)
        from tridep_ccml.eeg_all_candidates import frozen_heads as F, paired_replay as P
        from tridep_ccml.tridep_eegpt import train_eval as T
        from tridep_ccml.eeg_all_candidates import storage as S
        torch, np, C = F.torch, F.np, F.C
        require(torch.cuda.is_available(), 'Real supplemental qualification requires CUDA')
        record['runtime'] = {'python': sys.version, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
                             'device': torch.cuda.get_device_name(), 'precision': 'float32', 'autocast': False}
        def persist():
            replace_own_trace(path, record)
        def forbid_step(*args, **kwargs):
            record['optimizer_step_started'] += 1; persist()
            raise RuntimeError('Supplemental measurement forbids every optimizer.step call')
        for cls in (torch.optim.AdamW, torch.optim.Adamax):
            old_steps.append((cls, cls.step)); cls.step = forbid_step
        torch.cuda.synchronize(); record['baseline_allocated_bytes'] = torch.cuda.memory_allocated()
        record['baseline_reserved_bytes'] = torch.cuda.memory_reserved(); torch.cuda.reset_peak_memory_stats(); memory_started = True
        arrays = []
        for row in plan['feature_records']:
            source = checked(row['features'])
            if row['archive_key'] is None:
                array = np.load(source, allow_pickle=False)
            else:
                with np.load(source, allow_pickle=False) as bundle:
                    array = bundle[row['archive_key']].copy()
            require(array.shape == (60, plan['dimension']) and array.dtype == np.float32 and np.isfinite(array).all(), 'Train-only feature shape/dtype/values differ')
            arrays.append(array)
            record['loaded_feature_sources'].append(row['features'])
            record['feature_archive_keys_read'].append({'subject_id': row['subject_id'], 'key': row['archive_key']})
            persist()
        C.runtime(11)
        assert_terminal_still_bound(gate, out)
        record['model_constructions_started'] += 1; persist()
        model = (T.original_model() if plan['paired'] else F.SubjectHead(plan['dimension'])).to('cuda')
        record['model_constructions_returned'] += 1; persist()
        require([n for n, p in model.named_parameters() if p.requires_grad] == plan['contract']['trainable_parameter_names'], 'Actual head parameter names differ from locked engineering')
        require(sum(p.numel() for p in model.parameters()) == plan['contract'].get('parameters', plan['contract']['input_contract'].get('parameters')), 'Head parameter count differs')
        features = torch.as_tensor(np.stack(arrays), device='cuda'); arrays = None
        target = torch.tensor(plan['labels'], device='cuda', dtype=torch.long)
        require(features.shape[0] == len(plan['probe_subject_ids']), 'Unexpected probe people')
        initial = C.state_digest(model.state_dict()); model.train()
        assert_terminal_still_bound(gate, out)
        torch.cuda.synchronize(); forward_began = time.perf_counter()
        record['forward_started'] += 1; persist()
        logits = model(features)
        record['forward_returned'] += 1; persist()
        loss = torch.nn.functional.cross_entropy(logits, target)
        require(torch.isfinite(loss), 'Nonfinite supplemental loss')
        record['backward_started'] += 1; persist()
        loss.backward()
        record['backward_returned'] += 1; record['gradient_passes'] = record['backward_returned']; persist()
        torch.cuda.synchronize(); record['forward_backward_seconds'] = time.perf_counter() - forward_began
        require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()), 'Missing or nonfinite head gradient')
        require(C.state_digest(model.state_dict()) == initial, 'Zero-update supplemental probe changed head state')
        optimizer = (torch.optim.Adamax(model.parameters(), lr=.005, weight_decay=0.) if plan['paired']
                     else torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001))
        rng = P.rng_state()
        saved = {'state_dict': T._cpu(model.state_dict()), 'optimizer_state_dict': T._cpu(optimizer.state_dict()),
                 'rng': rng, 'scheduler_state_dict': None if plan['paired'] else {'policy': 'frozen', 'completed_epoch': 0},
                 'sampler_position': {'epoch_completed': 0, 'next_batch': 0}, 'source': record['source'],
                 'condition_source_lock': plan['condition_source_lock'], 'probe_subject_ids': plan['probe_subject_ids']}
        S.save_torch(probe_path, saved, immutable=True)
        restored = torch.load(probe_path, map_location='cpu', weights_only=False)
        model.load_state_dict(restored['state_dict'], strict=True); optimizer.load_state_dict(restored['optimizer_state_dict']); P.restore_rng(restored['rng'])
        require(C.state_digest(model.state_dict()) == initial and P.rng_equal(rng, P.rng_state()), 'Supplemental resume model/RNG roundtrip differs')
        require(optimizer.state_dict()['state'] == {} and record['optimizer_step_started'] == record['optimizer_step_returned'] == 0, 'Supplemental optimizer mutated')
        require(restored['scheduler_state_dict'] == saved['scheduler_state_dict'] and restored['sampler_position'] == saved['sampler_position'], 'Supplemental roundtrip control state differs')
        checked(plan['condition_source_lock']); checked(plan['feature_manifest']); checked(record['source'])
        record.update(status='complete', passed=True, resume_checked=True, initial_state_sha256=initial,
                      restored_state_sha256=C.state_digest(model.state_dict()), resume_probe=descriptor(probe_path),
                      head_parameters=sum(p.numel() for p in model.parameters()), actual_trainable_parameter_names=[n for n,p in model.named_parameters() if p.requires_grad],
                      actual_optimizer={'name': type(optimizer).__name__, 'param_groups': [{k:v for k,v in group.items() if k != 'params'} for group in optimizer.state_dict()['param_groups']]})
    except BaseException as error:
        record.update(status='failed', passed=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        for cls, old in old_steps:
            cls.step = old
        if memory_started:
            try:
                torch.cuda.synchronize()
                record['peak_allocated_bytes'] = int(torch.cuda.max_memory_allocated())
                record['peak_reserved_bytes'] = int(torch.cuda.max_memory_reserved())
            except BaseException as error:
                record.update(status='failed', passed=False, resource_finalization_error=str(error))
        record.update(finished_at=now(), seconds=time.perf_counter()-began)
        replace_own_trace(path, record)
        model = optimizer = features = target = logits = loss = saved = restored = None
        gc.collect()
        if torch is not None and memory_started:
            torch.cuda.empty_cache()
    require(record['passed'] and record['peak_allocated_bytes'] > 0 and record['forward_backward_seconds'] > 0, 'Supplemental measurement incomplete')
    return descriptor(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--run', action='store_true', help='Execute all12 only after full360/Q3-owner-release/E2 gate')
    group.add_argument('--check-only', action='store_true', help='Default: inspect gate; never execute supplementary model work')
    args = parser.parse_args(argv)
    gate = inspect_gate()
    gate_path = WORK / REPORT_REL / ('supplemental_head_resource_gate_' + str(time.time_ns()) + '.json')
    write_new(gate_path, {'source': descriptor(SOURCE), **gate, 'supplemental_gpu_execution_requested': args.run})
    if not args.run or gate['status'] != 'ready_after_full_e2_verification':
        print(json.dumps({'gate': descriptor(gate_path), 'status': gate['status'], 'reasons': gate.get('reasons', []), 'supplemental_model_work': 0}, ensure_ascii=False))
        return 2 if args.run else 0
    out = WORK / OUT_REL; registration = read(checked(gate['registration']))
    conditions = [r['condition_id'] for r in registration['configuration']['conditions'] if r['policy_key'] in ('diagnostic','frozen')]
    require(len(conditions) == 12 and len(set(conditions)) == 12, 'Exactly12 diagnostic/frozen head conditions required')
    completed = []
    with exclusive_measurement_lease(out):
        for condition in conditions:
            assert_terminal_still_bound(gate, out)
            plan = probe_plan(condition, registration)
            existing = [path for path in (out / 'supervised_engineering' / condition).glob('measurement_*.json')
                        if successful_existing(path, plan=plan, gate=gate, source=descriptor(SOURCE))]
            require(len(existing) <= 1, 'Multiple completed supplemental attempts need explicit cost review')
            if existing:
                checked(read(existing[0])['resume_probe']); completed.append(descriptor(existing[0])); continue
            completed.append(measure(plan, gate))
    summary = {'subtype': 'supplemental_head_resource_measurement_completion', 'passed': True,
               'completed_at': now(), 'source': descriptor(SOURCE), 'gate': descriptor(gate_path),
               'conditions': conditions, 'measurements': completed, 'optimizer_updates': 0,
               'historical_missing_measurements_backfilled': False}
    path = WORK / REPORT_REL / ('supplemental_head_resource_complete_' + str(time.time_ns()) + '.json')
    write_new(path, summary)
    print(json.dumps({'complete': descriptor(path), 'conditions': len(completed), 'optimizer_updates': 0}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
