"""Qualified diagnostic-only chunking; no tensor imports or training changes."""
from contextlib import contextmanager
import hashlib
import json
import math
import statistics
from pathlib import Path

WORK = Path(__file__).resolve().parents[2]
REGISTRY = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/registration.json'
REGISTRY_SHA = '27bf2972c08f77457959cf3cba5bc6056c320f223c2426af5da3ae88d55ce090'
F1 = WORK / 'tridep_ccml/eeg_all_candidates/finetune.py'
F1_SHA = 'cda19474654233c95072acae475174d9011bb0739be650a597194f04c4b0aa24'
F2_SHA = 'fa53183ea4927daed5dcec0204e8a28ab09940f589742debda884402581c1a8b'
SAFE_BUILDERS = {'stopped_training': 'output/analysis/eeg_all_candidates_v1_20261008/safe_stop_acceleration_v1.py',
                 'queue_exclusive_post_prepare': 'tridep_ccml/eeg_all_candidates/acceleration_prepare_v1.py'}
CHUNKS = (8, 16, 32, 60)


def require(value, message):
    if not value:
        raise ValueError(message)


def descriptor(path):
    path = Path(path).resolve()
    require(path.is_relative_to(WORK) and path.suffix in ('.json', '.py'), 'Only workspace JSON/source evidence may be opened')
    return {'path': path.relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read_bound(binding):
    require(type(binding) is dict and set(binding) == {'path', 'sha256'}, 'Expected exact evidence descriptor')
    path = (WORK / binding['path']).resolve()
    require(path.is_relative_to(WORK) and path.suffix == '.json', 'Expected workspace JSON evidence')
    data = path.read_bytes()
    require({'path': path.relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(data).hexdigest()} == binding, 'Qualification bytes differ')
    return json.loads(data.decode('utf-8-sig'))


def validate_roles(raw, split, labels):
    require(set(split) == {'train', 'val', 'test'}, 'Expected exact train/val/test roles')
    sets = {r: set(split[r]) for r in split}
    require(all(len(sets[r]) == len(split[r]) and sets[r] for r in split), 'Empty/duplicate participant role')
    require(not (sets['train'] & sets['val'] or sets['train'] & sets['test'] or sets['val'] & sets['test']), 'Participant roles overlap')
    require(set(raw) == sets['train'] | sets['val'], 'Diagnostics raw keys must be exactly training plus validation; never test')
    for sid, value in raw.items():
        require(tuple(value.shape) == (60, 19, 1000) and str(value.dtype) == 'float32', 'Expected unchanged all60-window float32 raw contract')
        require(sid in labels and labels[sid] in (0, 1), 'Missing/bad diagnostic label')


@contextmanager
def diagnostic_chunk(model, raw, split, labels, chunk):
    """Used by qualification and the runtime wrapper; restores even on failure."""
    require(type(chunk) is int and chunk in CHUNKS, 'Unsupported diagnostic chunk')
    require(type(model.window_chunk) is int and model.window_chunk == 8, 'Training/reload default must remain8')
    validate_roles(raw, split, labels)
    model.window_chunk = chunk
    try:
        yield
    finally:
        model.window_chunk = 8


def verify_qualification(binding, condition, split=None, *, allow_tolerance=False):
    """Read JSON and source hashes only; checkpoint/raw descriptors are not opened."""
    require(allow_tolerance is False, 'Tolerance-only qualification is not authorized')
    report = read_bound(binding)
    require(report.get('schema') == 'diagnostic_batching_qualification_v1' and report.get('passed') is True
            and report.get('condition') == condition, 'Missing actual successful qualification for this condition')
    require(report.get('registration') == {'path': REGISTRY.relative_to(WORK).as_posix(), 'sha256': REGISTRY_SHA}
            and descriptor(REGISTRY) == report['registration'], 'Fixed registration differs')
    require(report.get('diagnostic_wrapper_source') == descriptor(Path(__file__))
            and report.get('original_diagnostics_source') == {'path': F1.relative_to(WORK).as_posix(), 'sha256': F1_SHA}
            and descriptor(F1) == report['original_diagnostics_source'], 'Diagnostic wrapper/delegate source differs')
    require(report.get('delegated_f2_source') == {'path': F1.with_name('finetune_v2.py').relative_to(WORK).as_posix(), 'sha256': F2_SHA}
            and descriptor(F1.with_name('finetune_v2.py')) == report['delegated_f2_source'], 'Sealed F2 model builder differs')
    require(report.get('baseline_chunk') == 8 and report.get('optimizer_updates') == 0
            and type(report['optimizer_updates']) is int and report.get('backward_calls') == 0
            and type(report['backward_calls']) is int and report.get('test_forward_count') == 0
            and type(report['test_forward_count']) is int, 'Qualification must be zero-update/no-backward/no-test')
    registry = read_bound(report['registration'])
    require(type(report['fold']) is int and report['fold'] in range(1, 6) and type(report['seed']) is int and report['seed'] in (11, 22, 33), 'Strict fold/seed identity required')
    qualification_split = registry['folds']['fold_' + str(report['fold'])]
    require(report['role_ids'] == qualification_split, 'Qualification fold/roles differ')
    mode = report['weight_mode']
    require(mode in ('stable_formal_checkpoint', 'fresh_official_initialization'), 'Unknown qualification weight mode')
    allowed = qualification_split['train'] + (qualification_split['val'] if mode == 'stable_formal_checkpoint' else [])
    require(set(report['raw_sources']) == set(allowed), 'Qualification includes forbidden raw person')
    registered_raw = {r['subject_id']: r['raw'] for r in registry['raw_sources']}
    require(report['raw_sources'] == {sid: registered_raw[sid] for sid in allowed}, 'Qualification raw descriptors differ from original registration')
    if mode == 'fresh_official_initialization':
        require(report['fold'] == 1 and report['seed'] == 11 and report['probe_split'] == {
            'train': qualification_split['train'][:-1], 'val': qualification_split['train'][-1:],
            'test': qualification_split['val'] + qualification_split['test']}, 'Fresh probe must use originalfold1 training people only')
    else:
        require(report['probe_split'] == qualification_split, 'Stable-checkpoint probe roles differ')
    lock = read_bound(report['condition_source_lock'])
    require(lock['condition'] == condition and lock['registration'] == report['registration'], 'Condition source lock differs')
    require(report['condition_source_lock']['path'] == (REGISTRY.parent / 'engineering' / (condition + '.json')).relative_to(WORK).as_posix(), 'Wrong condition lock path')
    require(lock['configuration_sha256'] == hashlib.sha256(json.dumps(lock['contract'], sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest(), 'Condition contract hash differs')
    for source in lock['contract']['implementation_sources']:
        require(descriptor(WORK / source['path']) == source, 'Condition implementation source changed')
    require(report.get('actual_runtime', {}).get('device_type') == 'cuda' and report.get('actual_runtime', {}).get('torch'), 'Real CUDA qualification required')
    require(report.get('run_requested') is True and report.get('qualification_mode') == 'bitwise_only', 'Only explicitly run bitwise qualification is admissible')
    require(report.get('qualifier_source') == descriptor(WORK / 'output/analysis/eeg_all_candidates_v1_20261008/qualify_acceleration_20261009.py'), 'Qualifier source differs')
    safe = read_bound(report['safe_stop_proof'])
    require(safe['schema'] == 'acceleration_safe_stop_v1' and safe['passed'] is True and safe['gpu_idle'] is True
            and (safe['condition'], safe['seed'], safe['fold']) == (condition, report['seed'], report['fold'])
            and safe['registration'] == report['registration'], 'Safe GPU proof binding differs')
    require(safe.get('mode') in SAFE_BUILDERS and safe['source']['path'] == SAFE_BUILDERS[safe['mode']]
            and descriptor(WORK / safe['source']['path']) == safe['source'], 'Safety proof builder/source changed')
    queue_registration = REGISTRY.parent / 'continuation_queue_v4/registration.json'
    if queue_registration.exists():
        queue = json.loads(queue_registration.read_text(encoding='utf-8-sig'))
        source_values = queue['sources'].values() if type(queue['sources']) is dict else queue['sources']
        require(safe['source'] in source_values, 'Safety proof source is not bound by Q4 registration')
    if mode == 'stable_formal_checkpoint':
        require(report['checkpoints'] == safe.get('checkpoints', [safe['checkpoint']] if safe.get('checkpoint') else [])
                and report['train_history'] == safe['train_history'], 'Stable states/history differ from safety proof')
        read_bound(report['train_history'])
    else:
        require(not safe.get('checkpoints') and not safe.get('checkpoint') and report['train_history'] is None
                and len(report['checkpoints']) == 1, 'Fresh qualification cannot include formal states')
        initial = read_bound(report['checkpoints'][0])
        require(initial['kind'] == 'fresh_official_initialization' and initial['condition'] == condition
                and initial['seed'] == 11 and initial['fold'] == 1 and initial['original_fold_train_only'] is True
                and type(initial['optimizer_updates']) is int and initial['optimizer_updates'] == 0
                and len(initial['state_sha256']) == 64, 'Fresh initial-state evidence differs')
    trace = read_bound(report['attempt_trace'])
    require(trace.get('status') == 'complete' and all(type(trace.get(k)) is int and trace[k] == 0 for k in
            ('optimizer_updates', 'backward_calls', 'test_forward_count', 'forbidden_backward_attempts', 'forbidden_optimizer_attempts')),
            'Qualification trace did not complete with zero updates/backward/test')
    require(trace.get('schema') == 'diagnostic_batching_qualification_attempt_v1' and trace['gpu_stage_entered'] is True
            and trace.get('safe_stop_proof') == report['safe_stop_proof']
            and all(trace.get('source_bindings', {}).get(k) == report[k] for k in
                    ('qualifier_source', 'diagnostic_wrapper_source', 'original_diagnostics_source', 'delegated_f2_source')),
            'Actual trace proof/source binding differs')
    require(type(trace['seconds']) in (int, float) and math.isfinite(trace['seconds']) and trace['seconds'] > 0, 'Invalid whole-attempt resource observation')
    if split is not None:
        require(any(split == registered for registered in registry['folds'].values()), 'Runtime roles are not an original registered fold')
    chunk = report['selected_chunk']
    require(type(chunk) is int and chunk in CHUNKS[1:], 'Acceleration requires a qualified16/32/60 candidate')
    candidates = [c for c in report['candidates'] if c['chunk'] == chunk]
    require(len(candidates) == 1 and candidates[0]['qualified'] is True, 'Selected chunk did not qualify')
    chosen = candidates[0]
    require(type(report['baseline_median_wall_seconds']) in (int, float) and math.isfinite(report['baseline_median_wall_seconds'])
            and report['baseline_median_wall_seconds'] > 0, 'Invalid baseline resource observation')
    require(type(chosen['median_wall_seconds']) in (int, float) and math.isfinite(chosen['median_wall_seconds'])
            and 0 < chosen['median_wall_seconds'] < report['baseline_median_wall_seconds'], 'Selected candidate lacks measured speed improvement')
    for measurement in report['measurements']:
        for key in ('synchronized_wall_seconds', 'gpu_event_interval_ms_not_busy_time', 'peak_allocated_bytes'):
            require(type(measurement[key]) in (int, float) and math.isfinite(measurement[key]) and measurement[key] >= 0, 'Invalid real resource observation')
        require(all(measurement[k] is True for k in ('state_unchanged', 'rng_restored', 'modes_restored')), 'Diagnostic measurement invariant failed')
    require(trace['measurements'] == report['measurements'], 'Report measurement list differs from durable trace')
    require(type(report['measured_repeats']) is int and 2 <= report['measured_repeats'] <= 5, 'Invalid measured repeat count')
    expected_measurements = {(json.dumps(cp, sort_keys=True), chunk, rep) for cp in report['checkpoints']
                             for chunk in CHUNKS for rep in range(report['measured_repeats'] + 1)}
    actual_measurements = [(json.dumps(m['checkpoint'], sort_keys=True), m['chunk'], m['repetition']) for m in report['measurements']]
    require(len(actual_measurements) == len(set(actual_measurements)) and set(actual_measurements) == expected_measurements, 'Missing/duplicate candidate measurements')
    require(all(type(m['chunk']) is int and type(m['repetition']) is int and m['warmup'] is (m['repetition'] == 0)
                for m in report['measurements']), 'Measurement identity/warmup differs')
    medians = {chunk: statistics.median(m['synchronized_wall_seconds'] for m in report['measurements']
                                       if m['chunk'] == chunk and not m['warmup']) for chunk in CHUNKS}
    require(medians[8] == report['baseline_median_wall_seconds'], 'Baseline summary differs from actual observations')
    require(len(report['candidates']) == 3 and {c['chunk'] for c in report['candidates']} == set(CHUNKS[1:]), 'Candidate coverage differs')
    require(all(type(c['chunk']) is int and type(c['qualified']) is bool and c['median_wall_seconds'] == medians[c['chunk']]
                for c in report['candidates']), 'Candidate summary differs from actual observations')
    eligible = [c for c in report['candidates'] if c['qualified'] and c['median_wall_seconds'] < medians[8]]
    require(min(eligible, key=lambda c: (c['median_wall_seconds'], c['chunk']))['chunk'] == chunk, 'Selected candidate is not fastest eligible measured candidate')
    require(chosen['equivalence_level'] == 'bitwise', 'Tolerance-only results are not runtime qualification')
    require(chosen['comparisons'] and len(chosen['comparisons']) == len(report['checkpoints']), 'All stable weight states must be compared')
    require(len({json.dumps(row['checkpoint'], sort_keys=True) for row in chosen['comparisons']}) == len(report['checkpoints']), 'Duplicate comparison checkpoint')
    for row in chosen['comparisons']:
        require(row['passed'] is True and all(row[k] is True for k in ('policy_decisions_equal', 'state_unchanged', 'rng_restored', 'modes_restored')), 'Qualification invariant failed')
        require(row['checkpoint'] in report['checkpoints'], 'Comparison checkpoint identity differs')
        for key in ('logits_max_abs', 'ce_max_abs'):
            require(type(row[key]) in (int, float) and math.isfinite(row[key]) and row[key] >= 0, 'Invalid equivalence measurement')
        if chosen['equivalence_level'] == 'bitwise':
            require(row['bitwise_logits'] is True and row['logits_max_abs'] == row['ce_max_abs'] == 0, 'Bitwise claim differs from measurements')
    require(report['selection_rule'] == 'fastest_qualified_median_wall_no_validation_performance_ranking', 'Selection must use execution cost, not validation performance')
    return report


def make_diagnostics(original, *, qualification, condition, on_boundary=None):
    """Install this callable only in F3's isolated F1 diagnostics slot."""
    report = verify_qualification(qualification, condition)
    chunk = report['selected_chunk']

    def wrapped(model, raw, split, labels):
        verify_qualification(qualification, condition, split)
        if on_boundary is not None:
            on_boundary('diagnostics_started')
        try:
            with diagnostic_chunk(model, raw, split, labels, chunk):
                result = original(model, raw, split, labels)
        except BaseException:
            if on_boundary is not None:
                on_boundary('diagnostics_failed')
            raise
        if on_boundary is not None:
            on_boundary('diagnostics_returned')
        return result

    return wrapped
