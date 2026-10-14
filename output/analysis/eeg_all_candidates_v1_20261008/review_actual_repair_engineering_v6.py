"""Review actual immutable repair-engineering evidence; default does no work.

This reads actual report/control/source bytes only; no Tensor/model/raw array/
GPU operation or probability is performed. Complete current-source-chain review
and actual formal fitting remain separate requirements.
"""
import argparse
import json
from pathlib import Path
import sys
import time
SOURCE = Path(__file__).resolve()
WORK = SOURCE.parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import storage as S
from tridep_ccml.eeg_all_candidates import deterministic_repair_execution_v6 as D
from tridep_ccml.eeg_all_candidates import qualify_deterministic_repairs_v8 as Q

def require(ok, message):
    if not ok:
        raise ValueError(message)

def review(binding, condition):
    Q.check_pause()
    Q.check_boundary()
    proof = D.qualification(binding, condition)
    report = proof['observed_engineering']
    contract = proof['contract']
    checks = {}

    def check(name, value):
        require(value, 'Independent actual engineering review failed: ' + name)
        checks[name] = True
    check('zero_optimizer_four_actual_returns', report['optimizer_step_started'] == report['optimizer_step_returned'] == 0 and report['backward_started'] == report['backward_returned'] == 4)
    check('six_subject_forwards_48_windows', report['forward_started'] == report['forward_returned'] == 6 and report['encoder_windows_started'] == report['encoder_windows_returned'] == 48)
    check('fixed_training_person_roles', len(contract['probe_subject_ids']) == 1 and contract['probe_subject_ids'][0] == report['probe_person'])
    check('native_forward_and_repeated_gradient_source_evidence', report['original_native_forward_bitwise_equal'] is True and len(report['checks']) == 2 and all((x['repeat_bitwise_equal'] is True for x in report['checks'])))
    check('complete_zero_step_recovery_and_parameter_partition', report['zero_step_optimizer_and_rng_resume_verified'] is True and set(contract['trainable_parameter_names']).isdisjoint(contract['frozen_parameter_names']))
    check('source_marker_scope_not_full_paper_or_speed', report['performance_improvement_claimed'] is False and report['native_CUDA_gradient_equivalence_established'] is False)
    check('Mamba_order_change_explicit', report['accumulation_order_changed'] is (condition == D.REPAIR_CONDITIONS[1]))
    check('actual_primary_scope_and_driver_nested', type(report['primary_engineering_seconds']) in (int, float) and report['primary_engineering_seconds'] > 0 and (S.read(S.checked(proof['controller']))['controller_seconds_primary_cost'] is False))
    worker = S.read(S.checked(proof['worker_owner']))
    admission = S.read(S.checked(report['admission']))
    check('source_bound_preCUDA_native_and_empty_context_admission', worker['GPU_before_runtime']['context_count'] == admission['GPU_admission']['context_count'] == 0 and worker['native']['controller_lease']['path'].endswith('owner.json') and (worker['native']['controller_lease']['sha256'] == proof['released_owner']['sha256']) and worker['native']['allowed_host'])
    check('no_actual_test_forward_or_inference', report['outer_test_read'] is False and report['formal_fits'] == 0)
    return {'status': 'actual_repair_engineering_reviewed', 'passed': True, 'reviewer_source': S.desc(SOURCE), 'qualification_report': binding, 'condition': condition, 'controller': proof['controller'], 'released_owner': proof['released_owner'], 'worker_owner': proof['worker_owner'], 'qualification_source': report['source'], 'actual_checks': checks, 'numerical_order_change_acknowledged': contract['accumulation_order_changed'], 'original_CUDA_gradient_equivalence_established': False, 'performance_improvement_claimed': False, 'checkpoint_tensor_loads': 0, 'model_or_GPU_operations': 0, 'outer_probability_reads': 0, 'source_weight_and_engineering_probe_bytes_hashed_for_binding': True, 'new_formal_training_or_final_plan_complete': False, 'limitations': ['Native inventory samples have gaps; no clean performance comparison is claimed.', 'Private original forward reused; original CUDA backward equivalence not established.', 'EEGMamba external-pretraining participant overlap exclusion remains unknown.']}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--condition', choices=D.REPAIR_CONDITIONS)
    parser.add_argument('--report')
    parser.add_argument('--report-sha256')
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps({'status': 'preparation_only', 'actual_engineering_review_started': False, 'model_calls': 0, 'GPU_calls': 0, 'checkpoint_tensor_loads': 0, 'outer_probability_reads': 0}))
        return
    require(all((args.condition, args.report, args.report_sha256)), 'Exact real condition/report binding required')
    value = review({'path': args.report, 'sha256': args.report_sha256}, args.condition)
    target = SOURCE.parent / ('independent_actual_repair_engineering_review_' + str(time.time_ns()) + '.json')
    with target.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(json.dumps(S.desc(target)))
if __name__ == '__main__':
    main()
