"""Pure standard-library cost semantics and isolated ledger regressions."""
from __future__ import annotations
import contextlib
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
import time
import unittest

REPORT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('cost_v4_tested', REPORT / 'cost_aggregation_v4.py')
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)

def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream: json.dump(data, stream)
    return {'path': path.relative_to(C.WORK).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

class CostSemantics(unittest.TestCase):
    def test_numeric_zero_boolean_negative_nonfinite(self):
        self.assertTrue(C.numeric(0))
        for value in (False, True, -1, math.nan, math.inf, '3', None): self.assertFalse(C.numeric(value))

    def test_aliases_deduplicate_and_conflicts_are_unknown(self):
        value, used, issue = C.counter({'a': 5, 'b': 5}, ('a','b'))
        self.assertEqual((value, used, issue), (5,['a','b'],None))
        self.assertIsNone(C.counter({'a':5,'b':7}, ('a','b'))[0])
        self.assertIsNotNone(C.counter({'a':True}, ('a',))[2])

    def test_generic_forward_does_not_become_outer_test(self):
        n = C.normalized({'forward_started':9,'forward_returned':8})['observed_counters']
        self.assertEqual(n['model_forward_returned'], 8)
        self.assertEqual(n['model_forward_started_not_returned'], 1)
        self.assertIsNone(n['outer_test_forward_started'])
        self.assertIsNone(n['outer_test_forward_returned'])
        self.assertEqual(C.normalized({'outer_test_forward_returned':0})['observed_counters']['outer_test_forward_returned'],0)

    def test_returned_exceeds_started_is_recorded_problem(self):
        n = C.normalized({'optimizer_step_started':2,'optimizer_step_returned':3})
        self.assertIsNone(n['observed_counters']['optimizer_started_not_returned'])
        self.assertTrue(any(x['problem']=='returned exceeds started' for x in n['counter_issues']))

    def test_missing_time_unknown_and_recorded_zero_kept(self):
        self.assertIsNone(C.elapsed({})[0])
        self.assertEqual(C.elapsed({'seconds':0})[0],0)
        self.assertIsNone(C.elapsed({'seconds':False})[0])
        self.assertEqual(C.elapsed({'started_at':'2026-10-09T00:00:00+08:00','finished_at':'2026-10-09T00:00:05+08:00'})[0],5)
        self.assertIsNone(C.elapsed({'started_at':'bad','finished_at':'bad'})[0])

    def test_peak_is_explicit_bytes_max_not_generic_units(self):
        n = C.normalized({'peak_allocated_bytes':100,'peak_gpu_allocated_bytes':200,'peak_memory':9000})
        self.assertEqual(n['peak_allocated_bytes'],200)
        self.assertEqual(n['peak_memory_units_unresolved'],9000)
        self.assertIsNone(C.normalized({'peak_memory':3000})['peak_allocated_bytes'])

    def test_legacy_d0_resume_is_opaque(self):
        failure={'last_durable_epoch':0,'optimizer_updates':40,'updates_completed_this_process':40}
        self.assertIsNone(C.rollback_lower_bound(failure,[{'epoch':1,'optimizer_updates':40}])[0])
        failure['last_durable_epoch']=True
        self.assertIsNone(C.rollback_lower_bound(failure,[{'epoch':1,'optimizer_updates':40}])[0])

    def test_legacy_positive_boundary_requires_consistent_process_history(self):
        failure={'last_durable_epoch':3,'optimizer_updates':12,'updates_completed_this_process':4}
        history=[{'epoch':3,'optimizer_updates':10}]
        self.assertEqual(C.rollback_lower_bound(failure,history)[0],2)
        failure['updates_completed_this_process']=1
        self.assertIsNone(C.rollback_lower_bound(failure,history)[0])
        failure['updates_completed_this_process']=True
        self.assertIsNone(C.rollback_lower_bound(failure,history)[0])

    def test_test_runtime_classified_by_content(self):
        self.assertEqual(C.test_runtime_category({'tests_run':6,'real_cuda_input_role':'train people only','toy_cpu_optimizer_updates_per_complete_test_execution':6}), 'mixed_engineering_test_audit')
        self.assertEqual(C.test_runtime_category({'tests_run':6,'gpu_used':False}), 'cpu_or_synthetic_test_audit')
        self.assertEqual(C.test_runtime_category({'tests_run':6}), 'test_audit_runtime_unknown')
        self.assertFalse(C.is_test_audit({'draft_not_completion_certificate':True,'tests_run':6}))

    def test_runtime_install_evidence_unknown_time(self):
        d={'packages_before':{'torch':'2.5'},'packages_after':{'torch':'2.5','ninja':'1.13'},'added_packages':['ninja'],'install':{'returncode':0},'optimizer_updates':0}
        n=C.runtime_setup_evidence(d)
        self.assertEqual(n['package_version_changes'],{'ninja':{'before':None,'after':'1.13'}})
        self.assertIsNone(n['duration_seconds'])
        self.assertEqual(n['optimizer_updates'],0)

    def test_recovery_requires_valid_boundaries_and_preserved_snapshots(self):
        data={'kind':'user_authorized_resume_after_paused_process_exit','condition':'Demo','seed':22,'fold':4,'durable_epoch':26,'durable_optimizer_updates':104,'durable_backward_calls':702,'paused_epoch':27,'paused_acknowledged_optimizer_updates':108,'paused_acknowledged_backward_calls':729,'acknowledged_updates_not_in_checkpoint':4,'acknowledged_backward_calls_not_in_checkpoint':27}
        history=[{'epoch':26,'optimizer_updates':104,'backward_calls':702}]
        progress={'epoch':27,'last_durable_epoch':26,'optimizer_updates_completed':108,'backward_calls_completed':729}
        good=C.recovery_extra_cost(data,history,progress)
        self.assertTrue(good['boundary_verified'])
        self.assertEqual(good['acknowledged_optimizer_updates_outside_checkpoint'],4)
        self.assertEqual(good['acknowledged_backward_calls_outside_checkpoint'],27)
        self.assertIsNone(good['prior_partial_attempt_model_seconds'])
        self.assertFalse(good['overnight_suspension_in_training_time'])
        self.assertFalse(C.recovery_extra_cost(data,None,progress)['boundary_verified'])
        self.assertFalse(C.recovery_extra_cost({**data,'acknowledged_updates_not_in_checkpoint':5},history,progress)['boundary_verified'])
        self.assertFalse(C.recovery_extra_cost({**data,'durable_epoch':True},history,progress)['boundary_verified'])
        self.assertFalse(C.recovery_extra_cost(data,history,{**progress,'optimizer_updates_completed':109})['boundary_verified'])

    def test_complete_isolated_inventory_deduplicates_scopes_and_preserves_unknown(self):
        fixture = Path(tempfile.mkdtemp(prefix='cost_fixture_',dir=REPORT)).resolve()
        self.assertEqual(fixture.parent, REPORT.resolve())
        saved = C.OUT,C.REPORT
        try:
            C.OUT, C.REPORT = fixture/'out',fixture/'reports'
            C.REPORT.mkdir()
            condition='Demo-lastblock-finetune'
            fit_id=condition+'/seed_11/fold_1'
            fit={'fit_id':fit_id,'condition':condition,'seed':11,'fold':1,'objective':'supervised'}
            write(C.OUT/'registration.json',{'configuration':{'conditions':[{'condition_id':condition,'policy_key':'finetune'}]},'fits':[fit]})
            run=C.OUT/'runs'/fit_id
            md=write(run/'metadata.json',{'condition':condition,'seed':11,'fold':1,'epochs':2,'optimizer_updates':10,'peak_allocated_bytes':200})
            write(run/'status.json',{'status':'training_complete','evidence':{'metadata':md}})
            write(run/'train_history.json',[{'epoch':2,'optimizer_updates':10}])
            write(run/'failure_1.json',{'optimizer_updates':10,'last_durable_epoch':0,'updates_completed_this_process':10})
            trace={'kind':'formal','condition':condition,'seed':11,'fold':1,'started_at':'2026-10-09T01:00:00+08:00','status':'failed','optimizer_step_started':5,'optimizer_step_returned':4,'forward_started':9,'forward_returned':9,'peak_allocated_bytes':300}
            write(C.OUT/'finetune_v2_attempts'/condition/'a'/'attempt.json',trace)
            first={'started_at':'2026-10-09T01:01:00+08:00','status':'failed','started_windows':100,'returned_windows':100,'peak_allocated_bytes':400}
            second={**first,'started_at':'2026-10-09T01:02:00+08:00','status':'complete'}
            write(C.OUT/'frozen_features'/'demo'/'export_attempts'/'first.json',first)
            write(C.OUT/'frozen_features'/'demo'/'export_attempts'/'second.json',second)
            write(C.OUT/'ready_recovery'/'demo'/'export_attempts'/'copy.json',first)
            write(C.OUT/'assets/eegmamba_dependency_repair/ninja_v1/installation_1.json',{'created_at_unix_ns':'1','passed':True,'install':{'returncode':0},'packages_before':{},'packages_after':{'ninja':'1.13'},'optimizer_updates':0,'model_forwards':0})
            write(C.REPORT/'cpu_misleading_test_results.json',{'passed':True,'tests_run':6,'real_cuda_input_role':'train people only','real_candidate_optimizer_updates':0,'toy_cpu_optimizer_updates_per_complete_test_execution':6,'earlier_test_execution':{'seconds':5,'failures':1},'peak_allocated_bytes':99999})
            write(C.REPORT/'cpu_test_results.json',{'passed':True,'tests_run':6,'gpu_used':False,'synthetic_optimizer_updates':9,'total_measured_optimizer_updates_recorded':25})
            write(C.REPORT/'oof_probabilities.json',{'not_to_be_read':True})
            saved_history=write(C.OUT/'resume_events/1/paused_fit_train_history.json',[{'epoch':2,'optimizer_updates':10,'backward_calls':20}])
            saved_progress=write(C.OUT/'resume_events/1/paused_fit_progress.json',{'epoch':3,'last_durable_epoch':2,'optimizer_updates_completed':14,'backward_calls_completed':47})
            resume={'kind':'user_authorized_resume_after_paused_process_exit','condition':condition,'seed':11,'fold':1,'durable_epoch':2,'durable_optimizer_updates':10,'durable_backward_calls':20,'paused_epoch':3,'paused_acknowledged_optimizer_updates':14,'paused_acknowledged_backward_calls':47,'acknowledged_updates_not_in_checkpoint':4,'acknowledged_backward_calls_not_in_checkpoint':27,'pause_record':{'path':'synthetic/pause.json','sha256':'b'*64},'preserved_artifacts':[{'preserved_copy':item,'original_at_resume':{'sha256':item['sha256']}} for item in (saved_history,saved_progress)]}
            write(C.OUT/'resume_events/1.json',resume)
            write(C.OUT/'resume_events/2.json',resume)
            value=C.collect()
            self.assertEqual(value['recovery_acknowledged_optimizer_updates_known_subtotal'],4)
            self.assertEqual(value['recovery_acknowledged_backward_calls_known_subtotal'],27)
            self.assertEqual(len(value['acknowledged_checkpoint_external_recovery_work']),2)
            self.assertEqual(sum(r['duplicate_checkpoint_boundary'] for r in value['acknowledged_checkpoint_external_recovery_work']),1)
            self.assertTrue(all(r['boundary_verified'] for r in value['acknowledged_checkpoint_external_recovery_work']))
            self.assertTrue(all(r['uncheckpointed_diagnostic_forward_count'] is None for r in value['acknowledged_checkpoint_external_recovery_work']))
            self.assertEqual(value['formal_durable_updates_known_subtotal'],10)
            self.assertTrue(value['formal_fits'][0]['metadata_matches_bound_completion_sha'])
            self.assertEqual(value['category_observations']['formal_training_attempt']['counter_totals']['optimizer_returned']['known_observed_subtotal'],4)
            self.assertEqual(value['category_observations']['feature_export']['primary_invocations'],2)
            self.assertEqual(value['category_observations']['feature_export']['counter_totals']['encoder_windows_returned']['known_observed_subtotal'],200)
            self.assertEqual(sum(r['aggregation_role']=='duplicate_invocation_artifact' for r in value['cost_records']),1)
            failure=next(r for r in value['cost_records'] if r['category']=='legacy_failure')
            self.assertIsNone(failure['observed_counters']['optimizer_returned'])
            self.assertIsNone(failure['acknowledged_checkpoint_external_updates_lower_bound'])
            trace=next(r for r in value['cost_records'] if r['category']=='formal_training_attempt')
            self.assertEqual(trace['observed_counters']['model_forward_returned'],9)
            self.assertIsNone(trace['observed_counters']['outer_test_forward_returned'])
            mixed=next(r for r in value['cost_records'] if r['category']=='mixed_engineering_test_audit')
            self.assertEqual(mixed['test_cost_evidence']['earlier_test_execution']['seconds'],5)
            self.assertEqual(mixed['test_cost_evidence']['toy_cpu_optimizer_updates_per_complete_test_execution'],6)
            self.assertEqual(value['category_observations']['mixed_engineering_test_audit']['primary_invocations'],0)
            self.assertEqual(value['category_observations']['cpu_or_synthetic_test_audit']['primary_invocations'],0)
            install=next(r for r in value['cost_records'] if r['category']=='runtime_environment_repair')
            self.assertIsNone(install['seconds_observed'])
            self.assertEqual(install['runtime_setup_evidence']['install_returncode'],0)
            self.assertEqual(value['max_observed_peak_allocated_bytes'],400)
            self.assertIsNone(value['all_plan_physical_optimizer_returns_exact'])
            self.assertIsNone(value['all_plan_wall_seconds_exact'])
            self.assertFalse(value['input_read_errors'])
            self.assertFalse(any('oof_probabilities' in d['path'] for d in value['inputs']))
            # Subsequent replay advances logical fit history; prior lost work survives once.
            (run/'metadata.json').write_text(json.dumps({'condition':condition,'seed':11,'fold':1,'epochs':3,'optimizer_updates':14,'backward_calls':47}))
            replayed=C.collect()
            self.assertEqual(replayed['formal_durable_updates_known_subtotal'],14)
            self.assertEqual(replayed['recovery_acknowledged_optimizer_updates_known_subtotal'],4)
            self.assertEqual(replayed['recovery_acknowledged_backward_calls_known_subtotal'],27)
            self.assertIsNone(replayed['all_plan_physical_optimizer_returns_exact'])
        finally:
            C.OUT,C.REPORT=saved
            # Only this explicitly created synthetic fixture directory is removed.
            assert fixture.resolve().parent == REPORT.resolve() and fixture.name.startswith('cost_fixture_')
            shutil.rmtree(fixture)

if __name__=='__main__':
    started=time.perf_counter()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CostSemantics)
    names=[test.id() for test in suite]
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    value={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'seconds':time.perf_counter()-started,'checks':names,'test_output':stream.getvalue(),'pure_standard_library':True,'gpu_calls':0,'model_constructions':0,'optimizer_updates':0,'raw_signal_reads':0,'test_probability_reads':0,'formal_fits':0,'source_descriptors':[{'path':p.relative_to(C.WORK).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (Path(__file__),Path(C.__file__))],'scope':'Pure logic plus short-lived synthetic JSON ledgers; no actual training sources or existing reports changed.'}
    target=REPORT/('cost_v4_stdlib_regressions_'+str(time.time_ns())+'.json')
    write(target,value)
    print(json.dumps({'path':str(target),'passed':value['passed'],'tests_run':result.testsRun,'output':stream.getvalue() if not result.wasSuccessful() else 'all passed'}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
