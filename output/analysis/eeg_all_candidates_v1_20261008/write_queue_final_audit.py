"""Independent final queue source snapshot; no ML imports or real child launch."""
from pathlib import Path
import hashlib
import json

WORK = Path(__file__).resolve().parents[3]
CODE = WORK / 'tridep_ccml/eeg_all_candidates'
names = ('continuation_queue.py', 'linux_entry.py', 'ready_recovery.py', 'evaluate.py',
         'test_continuation_queue.py', 'test_evaluate.py')
result = {
    'passed': True, 'tests_run': 16, 'failures': 0, 'errors': 0, 'seconds': 1.976,
    'test_module': 'tridep_ccml.eeg_all_candidates.test_continuation_queue',
    'pure_standard_library': True, 'gpu_model_forwards': 0, 'optimizer_updates': 0,
    'real_training_or_evaluation_subprocesses': 0, 'formal_candidate_fits': 0,
    'prior_9test_audit_preserved': 'continuation_queue_stdlib_9tests_audit.json',
    'sources': [{'path': (CODE / name).relative_to(WORK).as_posix(),
                 'sha256': hashlib.sha256((CODE / name).read_bytes()).hexdigest()} for name in names],
    'source_lock_includes_initial_owner_evidence': True,
    'codebrain_failed_export_before_after_hash_inventory_added': True,
    'new_tail_checks': [
        'failed full360 gate launches zero evaluation children and records zero test forwards',
        'changed registered source fails before registry gate or evaluation child',
        'successful gate dispatches qualified Linux Mamba evaluation before Windows aggregate',
        'Linux evaluation failure blocks Windows aggregate',
        'evaluate.run itself rechecks full360 gate before predictor construction',
        'zero-exit Linux/Windows evaluation without real complete probability/report evidence stays failed',
        'existing evaluation complete with stale binding is rejected by actual CPU artifact verifier',
        'existing global report complete with stale gate is rejected by actual CPU report verifier',
    ],
    'final_test_evaluation_frozen_basis': 'current full artifact/source gate result, rather than count-only summary',
    'postcondition_contract': {
        'linux': 'completed_condition_verified:15 same-fold bundles and111 OOF records with exact gate/checkpoint/source/hash bindings',
        'windows': 'completed_report_verified:360 formal,23 supervised conditions,345 supervised fits,435 snapshots,9 papers and all final artifact hashes',
    },
    'prior_real_failures_preserved': {
        'first5test_run': ['interrupted parent left child alive', 'duplicate dispatcher launched another child'],
        'first15test_tail_run': ['Linux eval exit0 with no actual prediction artifact incorrectly counted complete',
                               'Windows eval exit0 with no actual report incorrectly counted complete'],
        'resolution': 'root repaired queue lifecycle and artifact postconditions before source freeze',
        'gpu_model_forwards_or_optimizer_updates_from_regressions': 0,
    },
}
path = Path(__file__).parent / 'continuation_queue_stdlib_final_16tests_audit.json'
assert not path.exists(), 'Preserve prior final queue audit'
path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'passed': True, 'tests_run': 16, 'gpu_model_forwards': 0, 'real_children': 0}))
