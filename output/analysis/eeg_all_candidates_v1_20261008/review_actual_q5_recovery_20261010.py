"""Independent JSON/source review; never opens tensors or queries processes."""
import hashlib
import json
from pathlib import Path
import sys
import time

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import lost_f3_closure_v1 as L, storage as S


def desc(path):
    path = Path(path).resolve()
    if not path.is_relative_to(WORK) or path.suffix not in ('.json', '.py'):
        raise ValueError('Only workspace JSON/source evidence may be read')
    return {'path': path.relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def run():
    entry = WORK / 'tridep_ccml/eeg_all_candidates/recover_q5_from_pause_v1.py'
    if desc(entry)['sha256'] != '93142aa08565a87f98de8d139d0830ed4984b02290b1dc61ebccd6dfa7ea019a':
        raise ValueError('Reviewed entry source changed')
    manifest = S.OUT / 'continuation_queue_v5/recovery_registration.json'
    manifest_binding = desc(manifest)
    if manifest_binding['sha256'] != 'ec1500ce2671600fca1588e41f14fc559e12300824aaa5c4a2625671f8d2ec9f':
        raise ValueError('Actual recovery manifest changed')
    checked_manifest = L.verify_recovery_registration(manifest_binding)
    closure_path = S.OUT / 'process_loss_closures/lost_f3_1791597116209422300.json'
    closure_binding = desc(closure_path)
    if closure_binding['sha256'] != 'e56bbb201c067bf0bab31c0d405520fb2860deb12796f088072bb30bdd73de65':
        raise ValueError('Actual closure changed')
    closure = L.read_json(closure_binding)
    record = L.read_json(closure['original_attempt'])
    verified = L.verify_lost_f3_attempt(closure['original_attempt'], record,
                                        {'condition': 'CSBrain-lastblock-finetune', 'seed': 33, 'fold': 4})
    if verified is None or verified['closure'] != closure_binding:
        raise ValueError('Actual closure not independently verified')
    test_path = S.REPORT / 'independent_recovery_entry_stdlib_tests_1791597271384024500.json'
    tests = S.read(test_path)
    if not (tests['passed'] is True and tests['tests_run'] == 15 and tests['failures'] == tests['errors'] == 0):
        raise ValueError('Independent fault/routing tests failed')
    if 'torch' in sys.modules:
        raise ValueError('Review unexpectedly imported Torch')
    value = {'schema': 'independent_actual_q5_recovery_review_v1',
             'passed_recovery_lineage_and_gates': True,
             'source': desc(entry), 'review_source': desc(Path(__file__)),
             'recovery_registration': manifest_binding, 'process_loss_closure': closure_binding,
             'closure_verifier': desc(Path(L.__file__)),
             'inherited_source_count': 76, 'actual_recovery_source_count': len(checked_manifest['value']['sources']),
             'stdlib_tests': desc(test_path), 'stdlib_test_methods': 15,
             'exclusive_retained_intent_precedes_manifest_or_closure': True,
             'duplicate_intent_manifest_or_closure_fail_closed': True,
             'strict_pause_controls_checked_before_archival': True,
             'independent_closure_and_second_absence_check_precede_stale_owner_unlink': True,
             'same_Q5_F3_E3_core_and_original360_registration': True,
             'no_invented_natural_CSBrain15_handoff': True,
             'actual_CPU5_135fit_405load_gate_routes_original_Q1_all360_test_gate': True,
             'original_CPU4_gate_not_called_by_recovery_route': True,
             'actual_closure_validation': verified,
             'scope_limitation': {
                 'source_line': 41,
                 'finding': 'Non-old-PID worker search also requires lunwen2 in CommandLine; a pure python -m tridep_ccml.eeg_all_candidates invocation can be omitted.',
                 'effect': 'Exact four old PID absence remains checked. This query alone cannot prove absence of every other workspace worker.',
                 'follow_up_requested': 'Root should retain a separate broad module-regex CIM inventory allowing only the exact current recovered Q5 identities.',
                 'current_other_worker_absence_independently_proven_by_this_review': False,
                 'review_did_not_modify_or_restart_active_execution': True},
             'GPU_operations': 0, 'model_constructions': 0, 'checkpoint_reads': 0,
             'raw_or_test_reads': 0, 'process_queries': 0, 'process_control': 0,
             'original_artifacts_modified': False, 'full_plan_completion_claimed': False}
    target = S.REPORT / ('independent_actual_q5_recovery_review_' + str(time.time_ns()) + '.json')
    S.save(target, value, immutable=True)
    print(json.dumps({'report': desc(target), 'passed_recovery_lineage_and_gates': True,
                      'process_scan_scope_limitation_disclosed': True}))


if __name__ == '__main__':
    run()
