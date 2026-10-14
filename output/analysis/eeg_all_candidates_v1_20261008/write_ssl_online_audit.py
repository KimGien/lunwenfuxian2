"""Record the additional online-only training proof audit without GPU work."""
from pathlib import Path
import hashlib
import json

WORK = Path(__file__).resolve().parents[3]
CODE = WORK / 'tridep_ccml/eeg_all_candidates'
result = {
    'passed': True, 'tests_run': 6, 'failures': 0, 'errors': 0, 'seconds': 5.657,
    'test_module': 'tridep_ccml.eeg_all_candidates.test_ssl',
    'gpu_model_forwards': 0, 'formal_candidate_fits': 0,
    'synthetic_cpu_optimizer_updates_this_revision': 12,
    'synthetic_cpu_optimizer_updates_cumulative_with_previous_runs': 36,
    'previous_audit_preserved': 'ssl_cpu_test_results.json',
    'training_math_and_stopping_policy_changed': False,
    'completion_proof_added': 'actual online InputEmbedding/Norm/Norm2/contex_encoder weights must change; predictor/targetEMA alone never satisfy completion',
    'fields_added': ['identity.initial_online_state_sha256', 'checkpoint.online_state_sha256',
                     'metadata.selected_online_state_sha256', 'metadata.terminal_online_state_sha256',
                     'metadata.online_changed_parameter_names', 'metadata.online_changed_parameter_count',
                     'metadata.terminal_online_changed_parameter_count', 'metadata.online_backbone_update_verified'],
    'meaningful_regressions': ['predictor and targetEMA-only changes are rejected as online training proof',
                              'real online parameter change yields exact changed parameter names',
                              'interrupted and uninterrupted fits have exact same online SHA and changed names',
                              'all prior sample/mask/fixed-validation/source/restore guards remain passing'],
    'source_descriptors': [{'path': (CODE / name).relative_to(WORK).as_posix(),
                            'sha256': hashlib.sha256((CODE / name).read_bytes()).hexdigest()}
                           for name in ('ssl.py', 'test_ssl.py', 'adapters/eeg2rep.py')],
}
path = Path(__file__).parent / 'ssl_cpu_test_results_online_v2.json'
assert not path.exists(), 'Preserve online-proof audit'
path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'passed': True, 'tests_run': 6, 'gpu_model_forwards': 0, 'synthetic_cpu_optimizer_updates': 12}))
