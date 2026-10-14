"""Record a source-proven scope correction without executing historical verifiers."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

SOURCE = Path(__file__).resolve()
WORK = SOURCE.parents[3]
REPORT = SOURCE.parent
MOD = WORK / 'tridep_ccml/eeg_all_candidates'
OUT = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'


def read(path):
    path = Path(path).resolve()
    if not path.is_relative_to(WORK) or path.suffix not in ('.json', '.py'):
        raise ValueError('This erratum only reads workspace JSON/source')
    raw = path.read_bytes()
    return (json.loads(raw.decode('utf-8-sig')) if path.suffix == '.json' else raw.decode('utf-8-sig'),
            {'path': path.relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(raw).hexdigest()})


def function_evidence(path, name, required):
    source, binding = read(path)
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    text = ast.get_source_segment(source, node)
    if any(value not in text for value in required):
        raise ValueError('Historical call chain differs: ' + name)
    return {'source': binding, 'function': name, 'line': node.lineno,
            'required_source_fragments': required}


def main():
    original, original_binding = read(REPORT / 'independent_actual_q5_recovery_review_1791597324607449500.json')
    if original_binding['sha256'] != '48d84089c5f7550feb5c8a2b52ad839830583971efdf4fc6f10afd2fd4d6fb12':
        raise ValueError('Original review differs')
    reviewed_source, source_binding = read(REPORT / 'review_actual_q5_recovery_20261010.py')
    if source_binding != original['review_source'] or 'L.verify_lost_f3_attempt' not in reviewed_source:
        raise ValueError('Original review source differs')
    chain = [
        function_evidence(MOD / 'lost_f3_closure_v1.py', 'verify_lost_f3_attempt', ['return validate_closure']),
        function_evidence(MOD / 'lost_f3_closure_v1.py', 'validate_closure', ['X.verify_observer_record']),
        function_evidence(MOD / 'execution_acceleration_v1.py', 'verify_observer_record', ['verify_execution_acceleration']),
        function_evidence(MOD / 'execution_acceleration_v1.py', 'verify_execution_acceleration', ['load_execution_acceleration']),
        function_evidence(MOD / 'execution_acceleration_v1.py', 'load_execution_acceleration', ['R._condition_lock(condition)']),
        function_evidence(MOD / 'registry.py', '_condition_lock', ['["weight_sources"]', 'checked(source)']),
        function_evidence(MOD / 'registry.py', 'checked', ['digest(resolved)']),
        function_evidence(MOD / 'registry.py', 'digest', ['open("rb")', 'hashlib.file_digest']),
    ]
    lock, lock_binding = read(OUT / 'engineering/CSBrain-lastblock-finetune.json')
    weights = [value for value in lock['contract']['weight_sources'] if Path(value['path']).suffix not in ('.py', '.json')]
    if len(weights) != 1 or weights[0]['sha256'] != 'cfcdf9dcb069905e5e02e92fe3d59bd3574e779bf554cc02cb701994a79e99cc':
        raise ValueError('Expected official weight provenance differs')
    stage, stage_binding = read(REPORT / 'CSBrain_full15_JSON_stage_review_v3_1791601874868214000.json')
    if stage['official_binary_weight_reads_this_review'] != 0 or not stage['passed']:
        raise ValueError('Replacement strict JSON/source stage differs')
    value = {
        'schema': 'historical_binary_hash_scope_erratum_v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'source': read(SOURCE)[1], 'affected_review': original_binding, 'affected_review_source': source_binding,
        'source_proven_transitive_call_chain': chain, 'condition_source_lock': lock_binding,
        'official_binary_weight_descriptors_carried_not_reopened_this_erratum': weights,
        'correction': 'The historical review called L/X, which invokes registry._condition_lock and hashes official binary weight bytes. Its JSON/source-only and unqualified checkpoint_reads=0 scope description was too broad. The completed-fit/resume .pt tensors were not reopened by that review, and no tensor deserialization occurred.',
        'historical_official_binary_weight_hash_read_occurred': True,
        'historical_binary_hash_read_call_count_not_instrumented': None,
        'historical_torch_loads': 0, 'historical_model_constructions': 0,
        'historical_GPU_calls': 0, 'historical_raw_feature_or_test_reads': 0,
        'historical_process_queries_or_controls': 0,
        'effect_on_findings': 'Recovery lineage, closure and counter conclusions remain supported. This corrects I/O scope only; it does not add optimizer/backward/tensor-load costs or change the known extra4/27, unknown tails, or completed CPU5 tensor-load counts.',
        'CPU5_source_and_report_tensor_load_counts_modified': False,
        'strict_JSON_source_stage_replacement': stage_binding,
        'old_sources_and_reports_preserved': True,
        'operations_this_erratum': {'JSON_and_Python_source_reads_only': True, 'binary_weight_reads': 0,
            'tensor_loads': 0, 'model_constructions': 0, 'GPU_calls': 0, 'raw_features_or_test_reads': 0,
            'X_L_project_verifier_execution': 0, 'process_queries_or_controls': 0},
        'full_plan_completion_claimed': False,
    }
    target = REPORT / ('historical_binary_hash_scope_erratum_' + str(time.time_ns()) + '.json')
    with target.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'artifact': read(target)[1], 'correction_scope': 'historical official-weight byte hashing only'}))


if __name__ == '__main__': main()
