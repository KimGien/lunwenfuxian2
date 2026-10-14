"""Standard-library gate regressions; no real evaluator, model or final output."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

REPORT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("_final_cost_v3_gate_test", REPORT / "finalize_full_plan_cost_v3.py")
F = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(F)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class GateTests(unittest.TestCase):
    def setUp(self):
        self.fixture = Path(tempfile.mkdtemp(prefix="cost_finalizer_fixture_", dir=REPORT)).resolve()
        self.assertEqual(self.fixture.parent, REPORT.resolve())
        self.saved = F.OUT, F.QUEUE, F.RESOURCE_RUNNER, F.RESOURCE_SHA256, F.EXPECTED_RESOURCE_MEASUREMENTS
        F.EXPECTED_RESOURCE_MEASUREMENTS = 0
        F.OUT = self.fixture / "out"
        F.QUEUE = F.OUT / "continuation_queue_v4"
        self.fits = [{"fit_id": f"Demo{c}/seed_{s}/fold_{k}", "condition": f"Demo{c}",
                      "seed": s, "fold": k, "objective": "self_supervised" if c == 23 else "supervised"}
                     for c in range(24) for s in (11, 22, 33) for k in range(1, 6)]
        write(F.OUT / "registration.json", {"fits": self.fits})
        for fit in self.fits:
            write(F.OUT / "runs" / fit["fit_id"] / "status.json", {"status": "training_complete"})
        self.queue = {"status": "all_training_complete", "test_evaluation_completed": True,
                      "test_evaluation_status": "complete", "test_evaluation_frozen": False,
                      "outcomes": {"synthetic_job": "complete"}}
        write(F.QUEUE / "registration.json", {"jobs": [{"id": "synthetic_job"}]})
        write(F.QUEUE / "status.json", self.queue)
        self.real_cpu_gate = F.completed_cpu_gate
        self.source_patch = patch.object(F, "final_source_gate", return_value=[])
        self.cpu_patch = patch.object(F, "completed_cpu_gate", return_value={"passed":True,"gate":F.desc(F.QUEUE/"registration.json"),"audit":F.desc(F.QUEUE/"status.json")})
        self.source_patch.start(); self.cpu_patch.start()
        self.completion = {"passed": True, "formal_fits_verified": 360, "supervised_conditions": 23,
                           "supervised_fits_evaluated": 345, "evaluated_checkpoints": 435, "candidate_papers_count": 9}

    def tearDown(self):
        F.OUT, F.QUEUE, F.RESOURCE_RUNNER, F.RESOURCE_SHA256, F.EXPECTED_RESOURCE_MEASUREMENTS = self.saved
        self.source_patch.stop(); self.cpu_patch.stop()
        assert self.fixture.resolve().parent == REPORT.resolve() and self.fixture.name.startswith("cost_finalizer_fixture_")
        shutil.rmtree(self.fixture)

    def no_evaluator_or_cost(self):
        return patch.multiple(F, verify_existing_evaluation=lambda: self.fail("Evaluator reached before cheap gates"),
                              collect_cost=lambda: self.fail("Cost collection reached before cheap gates"))

    def bound_completion(self):
        gate = {"passed": True, "registration": F.desc(F.OUT / "registration.json"), "formal_fits_verified": 360,
                "statuses": [F.desc(F.OUT / "runs" / fit["fit_id"] / "status.json") for fit in self.fits]}
        canonical = hashlib.sha256(json.dumps(gate, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        path = F.OUT / "evaluation" / ("test_gate_" + canonical + ".json")
        write(path, gate)
        write(F.OUT / "evaluation/complete.json", {**self.completion, "test_gate": F.desc(path)})
        return path

    def test_incomplete_training_stops_before_queue_or_evaluation_read(self):
        write(F.OUT / "runs" / self.fits[-1]["fit_id"] / "status.json", {"status": "training"})
        original = F.read
        def guard(path):
            self.assertFalse(Path(path).is_relative_to(F.QUEUE))
            self.assertFalse(Path(path).is_relative_to(F.OUT / "evaluation"))
            return original(path)
        with patch.object(F, "read", guard), self.no_evaluator_or_cost():
            result = F.finalize()
        self.assertEqual(result["status"], "pending")
        self.assertEqual(result["formal_status_counts"], {"training_complete": 359, "training": 1})
        self.assertFalse(result["outer_prediction_artifacts_read"])

    def test_queue_not_final_rejects_before_evaluation_read(self):
        write(F.QUEUE / "status.json", {**self.queue, "test_evaluation_completed": False})
        original = F.read
        def guard(path):
            self.assertFalse(Path(path).is_relative_to(F.OUT / "evaluation"))
            return original(path)
        with patch.object(F, "read", guard), self.no_evaluator_or_cost():
            result = F.finalize()
        self.assertEqual(result["status"], "pending")

    def test_owner_must_be_released_even_if_summary_complete(self):
        write(F.QUEUE / "owner.json", {"id": "synthetic-live-owner"})
        with self.no_evaluator_or_cost():
            result = F.finalize()
        self.assertEqual(result["status"], "pending")
        self.assertIn("owner", result["reasons"][0])

    def test_every_registered_queue_job_must_be_complete(self):
        write(F.QUEUE / "status.json", {**self.queue, "outcomes": {"synthetic_job": "failed"}})
        with self.no_evaluator_or_cost():
            result = F.finalize()
        self.assertEqual(result["status"], "pending")

    def test_completion_manifest_must_exist(self):
        with self.no_evaluator_or_cost():
            result = F.finalize()
        self.assertEqual(result["status"], "pending")
        self.assertIn("evaluation/complete.json", result["reasons"][0])

    def test_boolean_or_wrong_evaluation_counts_are_rejected(self):
        write(F.OUT / "evaluation/complete.json", {**self.completion, "formal_fits_verified": True})
        with self.no_evaluator_or_cost():
            result = F.finalize()
        self.assertEqual(result["status"], "pending")
        write(F.OUT / "evaluation/complete.json", {**self.completion, "evaluated_checkpoints": 434})
        with self.no_evaluator_or_cost():
            self.assertEqual(F.finalize()["status"], "pending")

    def test_ready_gate_still_requires_real_verifier_before_collect_or_write(self):
        self.bound_completion()
        result = F.preflight()
        self.assertEqual(result["status"], "ready_for_readonly_verification")
        self.assertFalse(result["completed_report_verifier_called"])
        class VerificationSentinel(Exception):
            pass
        with patch.object(F, "verify_existing_evaluation", side_effect=VerificationSentinel("stop before any real verification")) as verifier:
            with patch.object(F, "collect_cost", side_effect=AssertionError("collection must wait for verifier")):
                with self.assertRaises(VerificationSentinel):
                    F.finalize()
        self.assertEqual(verifier.call_count, 1)
        self.assertFalse(list(self.fixture.rglob("final_full_plan_cost_*.json")))

    def test_missing_or_changed_original_gate_cannot_be_recreated(self):
        write(F.OUT / "evaluation/complete.json", self.completion)
        with self.no_evaluator_or_cost():
            self.assertEqual(F.finalize()["status"], "pending")
        path = self.bound_completion()
        write(path, {"passed": True, "formal_fits_verified": 1})
        with self.no_evaluator_or_cost():
            result = F.finalize()
        self.assertEqual(result["status"], "pending")
        self.assertIn("no gate may be recreated", result["reasons"][0])

    def test_duplicate_or_wrong_objective_matrix_rejects(self):
        write(F.OUT / "registration.json", {"fits": self.fits[:-1] + [self.fits[0]]})
        with self.no_evaluator_or_cost():
            self.assertEqual(F.finalize()["status"], "pending")

    def test_report_keeps_nulls_recovery_scope_and_explicit_optimizer_wording(self):
        # Only render a string in memory. No synthetic final report is written.
        snapshot = {"captured_at": "synthetic", "formal_durable_updates_known_subtotal": 100,
            "condition_cost_summaries": [], "category_observations": {"synthetic_attempt": {
                "primary_invocations": 1, "supplemental_records": 0,
                "known_recorded_scope_seconds_subtotal": None, "seconds_unknown_records": 1,
                "counter_totals": {"optimizer_returned": {"known_observed_subtotal": 4, "records_unknown": 0}}}}, "feature_manifest_cost_evidence": [],
            "recovery_acknowledged_optimizer_updates_known_subtotal": 4,
            "recovery_acknowledged_backward_calls_known_subtotal": 27,
            "max_observed_peak_allocated_bytes": None, "unknown_or_opaque_records": [], "rules": []}
        F.disclose_gaps({**snapshot, "cost_records": []})
        snapshot["cost_records"] = []
        F.disclose_gaps(snapshot)
        rendered = F.render_report(snapshot)
        self.assertRegex(rendered, r"(?<!\d)4\s*(次)?\s*(优化器|optimizer)")
        self.assertRegex(rendered, r"(?<!\d)27\s*(次)?\s*(反向|backward)")
        for term in ("checkpoint", "未知", "null", "暂停过夜不计训练", "诊断前向", "旧退出/耗时"):
            self.assertIn(term, rendered)
        self.assertIsNone(snapshot["all_plan_physical_optimizer_returns_exact"])
        self.assertIsNone(snapshot["all_plan_wall_seconds_exact"])
        self.assertIn("4（缺 0）", rendered)

    def test_missing_new_head_resource_measurement_stops_before_evaluator(self):
        registration = {"fits": self.fits, "configuration": {"conditions": [{"condition_id": "Demo0", "policy_key": "frozen"}]}}
        write(F.OUT / "registration.json", registration)
        write(F.OUT / "engineering/Demo0.json", {"contract": {"engineering_gradient_passes": 1}})
        write(F.OUT / "evaluation/complete.json", self.completion)
        with self.no_evaluator_or_cost():
            result = F.finalize()
        self.assertEqual(result["status"], "pending")
        self.assertEqual(result["supplemental_head_resource_gate"]["conditions_requiring_new_measurement"], ["Demo0"])

    def resource_fixture(self):
        F.EXPECTED_RESOURCE_MEASUREMENTS = 1
        people = [f'train_{n}' for n in range(8)]
        for fit in self.fits:
            fit.update(train_ids=people, val_ids=['val'], test_ids=['test'])
        registration = {"fits": self.fits, "configuration": {"conditions": [{"condition_id": "Demo0", "policy_key": "frozen"}]},
                        "folds": {"fold_1": {"train": people}}}
        write(F.OUT / "registration.json", registration)
        source = self.fixture / "synthetic_expected_resource_runner.py"
        source.write_text("# expected fixture source only; never executed\n", encoding="utf-8")
        F.RESOURCE_RUNNER = source
        F.RESOURCE_SHA256 = F.desc(source)['sha256']
        features = []
        for person in people:
            path=self.fixture/(person+'.bin');path.write_bytes(b'synthetic feature bytes; never loaded')
            features.append({'subject_id':person,'features':F.desc(path)})
        manifest_path=F.OUT/'frozen_features/Demo0/manifest.json'
        write(manifest_path, {'passed':True,'condition':'Demo0','encoder_fit_subject_ids':[],'records':features})
        contract={'engineering_gradient_passes':1,'probe_subject_ids':people,'implementation_sources':[F.desc(source)],
                  'weight_sources':[F.desc(manifest_path)],'input_contract':{'parameters':100},'trainable_parameter_names':['head.weight']}
        contract_sha=hashlib.sha256(json.dumps(contract,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        lock_path=F.OUT/'engineering/Demo0.json'
        write(lock_path,{'condition':'Demo0','registration':F.desc(F.OUT/'registration.json'),'configuration_sha256':contract_sha,'contract':contract})
        self.bound_completion()
        cpu_proof=self.fixture/'cpu_audit_synthetic.json';write(cpu_proof,{'synthetic':True})
        cpu_gate=F.QUEUE/'pre_test_finetune_cpu_gate.json';write(cpu_gate,{'audit':F.desc(cpu_proof)})
        probe_path=F.OUT/'supervised_engineering/Demo0/probe_synthetic.pt'
        probe_path.parent.mkdir(parents=True,exist_ok=True);probe_path.write_bytes(b'synthetic checkpoint identity only; never loaded')
        value={"passed": True, "status":"complete", "subtype": "supplemental_head_resource_measurement", "condition": "Demo0",
               "new_measurement_not_historical_backfill": True, "historical_resource_fields_modified":False,
               "registration": F.desc(F.OUT / "registration.json"), "condition_source_lock": F.desc(lock_path),
               "source": F.desc(source), "delegated_resource_measurement_source":F.desc(F.RESOURCE_DELEGATE), "resource_measurement_version":3, "seconds": 0.5, "forward_backward_seconds":0.25,
               "peak_allocated_bytes": 1024, "optimizer_updates": 0, "gradient_passes":1, "input_role": "train",
               "seed": 11, "fold": 1, "probe_subject_ids": people, 'feature_manifest':F.desc(manifest_path),
               'loaded_feature_sources':[r['features'] for r in features],
               'feature_archive_keys_read':[{'subject_id':person,'key':None} for person in people],
               'actual_trainable_parameter_names':['head.weight'], 'optimizer_step_started':0,'optimizer_step_returned':0,
               'formal_fits':0,'new_encoder_forwards':0,'forward_started':1,'forward_returned':1,'backward_started':1,'backward_returned':1,
               'model_constructions_started':1,'model_constructions_returned':1,'resume_checked':True,
               'initial_state_sha256':'a'*64,'restored_state_sha256':'a'*64,'resume_probe':F.desc(probe_path),'head_parameters':100,
               'actual_optimizer':{'name':'AdamW'}, 'measurement_scope':{k:'synthetic scope' for k in ('seconds','forward_backward_seconds','peak_allocated_bytes','features','no_historical_backfill')},
               'gate':{'status':'ready_after_full_e3_verification','registration':F.desc(F.OUT/'registration.json'),
                       'q4_registration':F.desc(F.QUEUE/'registration.json'),'q4_terminal_status':F.desc(F.QUEUE/'status.json'),
                       'pre_test_finetune_cpu_verification':{'gate':F.desc(cpu_gate),'audit':F.desc(cpu_proof),
                           'auditor_source':F.desc(F.CPU_AUDITOR),'actual_cpu_checkpoint_loads':405,'actual_verified_finetune_identities':135},
                       'evaluation_verification':{'passed':True,'complete_manifest':F.desc(F.OUT/'evaluation/complete.json'),
                           'formal_fits_verified':360,'supervised_fits_evaluated':345,'supervised_conditions':23,
                           'model_constructions':0,'gpu_forward_calls':0,
                           'evaluator_execution':{'version':3,'wrapper_source':F.desc(F.WORK/'tridep_ccml/eeg_all_candidates/evaluate_v3.py')}}}}
        path=F.OUT/'supervised_engineering/Demo0/measurement_synthetic.json'
        write(path,value)
        return registration,path,value

    def test_actual_resource_trace_scope_and_current_bindings_required(self):
        registration,path,value=self.resource_fixture()
        result=F.head_resource_gate(registration)
        self.assertTrue(result['passed'],str(result))
        self.assertTrue(result['verified_measurements'][0]['old_missing_seconds_remains_unknown'])
        self.assertTrue(result['verified_measurements'][0]['old_missing_peak_remains_unknown'])
        foreign=self.fixture/'foreign_source.py';foreign.write_text('# wrong actual source',encoding='utf-8')
        changes=({'status':'failed'}, {'optimizer_step_started':1,'optimizer_step_returned':1}, {'formal_fits':1},
                 {'optimizer_updates':1}, {'probe_subject_ids':list(reversed(value['probe_subject_ids']))},
                 {'probe_subject_ids':['test']}, {'new_measurement_not_historical_backfill':False},
                 {'historical_resource_fields_modified':True}, {'seconds':None}, {'peak_allocated_bytes':True},
                 {'input_role':'test'}, {'gradient_passes':0}, {'forward_backward_seconds':None},
                 {'forward_backward_seconds':float('nan')}, {'forward_returned':0}, {'backward_started':True},
                 {'model_constructions_returned':0}, {'new_encoder_forwards':1}, {'resume_checked':False},
                 {'restored_state_sha256':'b'*64}, {'source':F.desc(foreign)}, {'head_parameters':99},
                 {'seed':22}, {'fold':2}, {'actual_optimizer':{'name':'Adamax'}},
                 {'feature_manifest':F.desc(foreign)}, {'loaded_feature_sources':[]},
                 {'feature_archive_keys_read':[{'subject_id':p,'key':'wrong'} for p in value['probe_subject_ids']]},
                 {'actual_trainable_parameter_names':[]}, {'gate':{**value['gate'],'q4_terminal_status':F.desc(foreign)}})
        for change in changes:
            with self.subTest(change=change):
                write(path,{**value,**change})
                self.assertFalse(F.head_resource_gate(registration)['passed'])
        write(path,value)
        missing_probe=F.checked(value['resume_probe']);missing_probe.write_bytes(b'changed checkpoint bytes')
        self.assertFalse(F.head_resource_gate(registration)['passed'])

    def test_supplemental_owner_blocks_even_complete_resource_record(self):
        registration,path,value=self.resource_fixture()
        write(F.OUT/'supervised_engineering/supplemental_head_resource_owner.json',{'id':'still-alive'})
        with self.no_evaluator_or_cost():
            result=F.finalize()
        self.assertEqual(result['status'],'pending')
        self.assertIn('owner',result['supplemental_head_resource_gate']['pending'][0]['reason'])

    def test_current_e3_gate_and_single_resource_attempt_are_required(self):
        registration,path,value=self.resource_fixture()
        wrong={**value['gate']['evaluation_verification'],'formal_fits_verified':359}
        write(path,{**value,'gate':{**value['gate'],'evaluation_verification':wrong}})
        self.assertFalse(F.head_resource_gate(registration)['passed'])

    def cpu_fixture(self):
        for index, fit in enumerate(self.fits):
            fit['stage'] = '3' if index < 135 else '2'
        registration={'fits':self.fits}
        write(F.OUT/'registration.json',registration)
        rows=[]
        for fit in self.fits[:135]:
            folder=F.OUT/'runs'/fit['fit_id']
            write(folder/'metadata.json',{'condition':fit['condition'],'seed':fit['seed'],'fold':fit['fold']})
            checkpoints=[]
            for name in ('initial.pt','latest.pt','best.pt'):
                (folder/name).write_bytes(b'synthetic identity only; no tensor load')
                checkpoints.append(F.desc(folder/name))
            complete={'metadata':F.desc(folder/'metadata.json'),'checkpoint':checkpoints[2]}
            write(folder/'complete.json',complete)
            write(folder/'status.json',{'status':'training_complete','evidence':complete})
            rows.append({'fit_id':fit['fit_id'],'condition':fit['condition'],'passed':True,
                'status_artifact':F.desc(folder/'status.json'),'completion':F.desc(folder/'complete.json'),
                'metadata':complete['metadata'],'cpu_checkpoint_audit':{'cpu_tensors_verified':True,'only_map_location_cpu':True,
                'checkpoint_loads':3,'gpu_operations':0,'checkpoints':checkpoints}})
        module_root=F.WORK/'tridep_ccml/eeg_all_candidates'
        delegates={key:F.desc(path) for key,path in {
            'wide_acceptance':F.REPORT/'audit_full_plan_acceptance_v3.py','finetune_v2':module_root/'finetune_v2.py',
            'finetune_v1':module_root/'finetune.py','registry':module_root/'registry.py','protocol':module_root/'protocol.py',
            'storage':module_root/'storage.py','execution_acceleration':module_root/'execution_acceleration_v1.py',
            'previous_cpu_auditor':F.REPORT/'audit_completed_finetune_checkpoints_v3.py'}.items()}
        proof={'auditor_source':F.desc(F.CPU_AUDITOR),'registration':F.desc(F.OUT/'registration.json'),
            'schema_version':4,'registered_total_fits':360,'registered_finetune_fits':135,'registered_finetune_conditions':9,
            'training_complete_finetune_observed':135,'verified_completed_finetune_fits':135,'logical_completed_fits_counted':135,
            'checkpoint_loads':405,'model_constructions':0,'gpu_operations':0,'non_finetune_checkpoints_read':0,
            'status':'all_finetune_verified','all_135_finetune_checkpoints_verified':True,'partial_snapshot':False,
            'finetune_status_counts':{'training_complete':135},'errors':[],'pending_fits':[],'deferred_completed_fits':[],
            'raw_arrays_loaded':False,'outer_prediction_artifacts_read':False,'test_gate_called':False,'evaluation_driver_called':False,
            'observed_auditor_operations_including_failed_checks':{'cpu_checkpoint_loads_started':405,
            'cpu_checkpoint_loads_returned':405,'model_construction_attempts_blocked':0,'gpu_initialization_attempts_blocked':0},
            'completed_rows':rows,'delegated_sources':delegates}
        path=F.QUEUE/'pre_test_finetune_cpu_audit_synthetic.json'
        def persist(value):
            write(path,value)
            write(F.QUEUE/'pre_test_finetune_cpu_gate.json',{'passed':True,'audit':F.desc(path),'checkpoint_loads':405,'test_forward_calls_this_check':0})
        persist(proof)
        return registration,proof,persist

    def test_cpu_gate_requires_exact135_and405_with_no_partial_or_errors(self):
        registration,proof,persist=self.cpu_fixture()
        self.assertTrue(self.real_cpu_gate(registration)['passed'])
        for change in ({'checkpoint_loads':404},{'checkpoint_loads':True},{'verified_completed_finetune_fits':134},
                       {'partial_snapshot':True},{'errors':[{'failed':True}]},{'deferred_completed_fits':[{}]},
                       {'completed_rows':proof['completed_rows'][:-1]+[proof['completed_rows'][0]]},
                       {'delegated_sources':{}}):
            with self.subTest(change=list(change)):
                persist({**proof,**change})
                with self.assertRaises(ValueError):self.real_cpu_gate(registration)

    def test_cpu_gate_rejects_changed_checkpoint_bytes_without_deserialization(self):
        registration,proof,persist=self.cpu_fixture()
        binding=proof['completed_rows'][0]['cpu_checkpoint_audit']['checkpoints'][0]
        (F.WORK/binding['path']).write_bytes(b'changed bytes')
        with self.assertRaisesRegex(ValueError,'changed'):self.real_cpu_gate(registration)

    def test_missing_cpu_proof_blocks_before_e3_after_other_gates(self):
        self.bound_completion()
        with patch.object(F,'completed_cpu_gate',side_effect=ValueError('missing actual CPU audit')),self.no_evaluator_or_cost():
            result=F.finalize()
        self.assertEqual(result['status'],'pending')
        self.assertIn('135-fit/405',result['reasons'][0])

    def test_source_mismatch_stops_before_e3_or_resource_reads(self):
        with patch.object(F,'final_source_gate',side_effect=ValueError('changed source')),self.no_evaluator_or_cost():
            result=F.finalize()
        self.assertEqual(result['status'],'pending')
        self.assertIn('source',result['reasons'][0])

    def test_old_q3_owner_also_blocks_finalization(self):
        write(F.OUT/'continuation_queue_v3/owner.json',{'id':'must remain blocked'})
        with self.no_evaluator_or_cost():result=F.finalize()
        self.assertEqual(result['status'],'pending')
        self.assertIn('Q3 owner',result['reasons'][0])

    def test_new_resource_version_and_pre_test_cpu_binding_are_required(self):
        registration,path,value=self.resource_fixture()
        for change in ({'resource_measurement_version':2},{'resource_measurement_version':True},
                       {'delegated_resource_measurement_source':value['source']},
                       {'gate':{**value['gate'],'pre_test_finetune_cpu_verification':{}}}):
            write(path,{**value,**change})
            with self.subTest(change=change):self.assertFalse(F.head_resource_gate(registration)['passed'])
        write(path,value)
        write(path.with_name('measurement_duplicate.json'),value)
        self.assertFalse(F.head_resource_gate(registration)['passed'])


class MambaBindingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = Path(tempfile.mkdtemp(prefix="cost_mamba_fixture_", dir=REPORT)).resolve()
        old = self.fixture / 'old.json'; recovered = self.fixture / 'recovered.json'; provenance = self.fixture / 'provenance.json'
        write(old, {'seconds':140.135474809}); write(recovered, {'seconds':140.135474809}); write(provenance, {'synthetic_fixture':True})
        self.probe = self.fixture / 'probe.json'
        write(self.probe, {'passed':True,'seconds':20.169247279,'peak_gpu_allocated_bytes':309073408,
                          'gpu_encode_calls':4,'engineering_optimizer_steps':1,'formal_fits':0})
        self.ledger = self.fixture / 'ledger.json'
        write(self.ledger, {'passed':True,'status':'complete','engineering':True,'cohort_data_or_labels_used':False,
            'encode_started':4,'encode_returned':4,'backward_started':1,'backward_returned':1,'sgd_step_started':1,'sgd_step_returned':1,
            'formal_fits':0,'formal_optimizer_updates':0,'started_at':'2026-10-09T20:11:30.770496+08:00',
            'finished_at':'2026-10-09T20:11:52.498263+08:00'})
        self.proof = self.fixture / 'proof.json'
        write(self.proof, {'passed':True,'real_new_recheck_report_archived':F.desc(self.probe),
            'new_recheck_primary_ledger':F.desc(self.ledger),'original_restored_report':F.desc(old),
            'original_exact_recovered_copy':F.desc(recovered),'recovery_provenance':F.desc(provenance),
            'new_recheck_is_engineering_not_formal_training':True,'old_source_bound_reports_and_new_real_cost_both_preserved':True,
            'original_report_unchanged_after_exact_byte_recovery':True,'nested_time_and_counter_reports_are_not_added_again':True,
            'no_model_checkpoint_or_raw_array_operations':True,'new_measured_seconds':20.169247279,
            'old_measured_seconds':140.135474809,'new_peak_allocated_bytes':309073408})
        self.constants = patch.multiple(F, KNOWN_MAMBA_RESTORE=F.desc(self.proof), KNOWN_MAMBA_ARCHIVE=F.desc(self.probe),
            KNOWN_MAMBA_LEDGER=F.desc(self.ledger), KNOWN_MAMBA_OLD_SHA256=F.desc(old)['sha256'])
        self.constants.start()
        self.primary = {'artifact':F.desc(self.ledger),'category':'mamba_gpu_engineering','aggregation_role':'primary_observed_invocation',
            'counter_issues':[],'observed_counters':{'encoder_forward_started':4,'encoder_forward_returned':4,'backward_started':1,
                'backward_returned':1,'optimizer_started':1,'optimizer_returned':1},'seconds_observed':21.727767,
            'seconds_source_field':'finished_at minus started_at'}
        self.snapshot = {'cost_records':[self.primary],'inputs':[], 'max_observed_peak_allocated_bytes':400000000,
            'category_observations':{'mamba_gpu_engineering':{'primary_invocations':1,'counter_totals':{'optimizer_returned':1},
                'known_recorded_scope_seconds_subtotal':21.727767}}, 'formal_durable_updates_known_subtotal':999}

    def tearDown(self):
        self.constants.stop()
        assert self.fixture.parent == REPORT.resolve() and self.fixture.name.startswith('cost_mamba_fixture_')
        shutil.rmtree(self.fixture)

    def test_inner_probe_binds_without_adding_primary_or_formal_cost(self):
        before = json.loads(json.dumps(self.snapshot['category_observations']['mamba_gpu_engineering']))
        F.bind_mamba_recheck_evidence(self.snapshot)
        self.assertEqual(self.snapshot['category_observations']['mamba_gpu_engineering'],before)
        self.assertEqual(self.snapshot['formal_durable_updates_known_subtotal'],999)
        self.assertEqual(self.snapshot['max_observed_peak_allocated_bytes'],400000000)
        self.assertEqual(self.snapshot['mamba_recheck_cost_evidence']['inner_probe_seconds'],20.169247279)
        self.assertFalse(self.snapshot['mamba_recheck_cost_evidence']['nested_seconds_added_again'])
        self.assertIsNone(self.snapshot['category_observations']['mamba_recheck_inner_probe']['known_recorded_scope_seconds_subtotal'])
        with self.assertRaises(ValueError): F.bind_mamba_recheck_evidence(self.snapshot)

    def test_new_probe_peak_is_max_not_sum_and_unknown_fulltime_stays_null(self):
        self.snapshot['max_observed_peak_allocated_bytes'] = None
        F.bind_mamba_recheck_evidence(self.snapshot)
        self.assertEqual(self.snapshot['max_observed_peak_allocated_bytes'],309073408)
        self.assertIsNone(self.snapshot['mamba_recheck_cost_evidence']['full_invocation_precise_wall_seconds'])
        self.assertIsNone(self.snapshot['mamba_recheck_cost_evidence']['gpu_busy_seconds'])

    def test_missing_duplicate_or_wrong_primary_counter_is_rejected(self):
        for records in ([], [self.primary,self.primary], [{**self.primary,'aggregation_role':'supplemental'}],
                        [{**self.primary,'observed_counters':{**self.primary['observed_counters'],'optimizer_returned':True}}],
                        [{**self.primary,'seconds_observed':20.169247279}]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                F.bind_mamba_recheck_evidence({**self.snapshot,'cost_records':records})

    def test_changed_or_unrelated_archive_is_rejected_without_snapshot_mutation(self):
        before = json.loads(json.dumps(self.snapshot))
        write(self.probe, {**F.read(self.probe),'engineering_optimizer_steps':2})
        with self.assertRaises(ValueError): F.bind_mamba_recheck_evidence(self.snapshot)
        self.assertEqual(self.snapshot,before)
        changed = F.desc(self.probe)
        with patch.object(F,'KNOWN_MAMBA_ARCHIVE',changed), self.assertRaises(ValueError):
            F.bind_mamba_recheck_evidence(self.snapshot)
        self.assertEqual(self.snapshot,before)


if __name__ == "__main__":
    started = time.perf_counter()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in (GateTests,MambaBindingTests))
    names = [test.id() for group in suite for test in group]
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    value = {"passed": result.wasSuccessful(), "tests_run": result.testsRun, "failures": len(result.failures),
             "errors": len(result.errors), "seconds": time.perf_counter()-started, "tests": names,
             "test_output": stream.getvalue(), "pure_standard_library": True,
             "real_completed_report_verifier_calls": 0, "gpu_calls": 0, "model_constructions": 0,
             "raw_signal_reads": 0, "test_probability_reads": 0, "optimizer_updates": 0,
             "final_cost_artifacts_created": 0,
             "sources": [F.desc(Path(__file__)), F.desc(Path(F.__file__))]}
    target = REPORT / ("final_cost_v3_gate_stdlib_tests_" + str(time.time_ns()) + ".json")
    with target.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"passed": value["passed"], "tests_run": result.testsRun, "evidence": F.desc(target),
                      "output": stream.getvalue() if not result.wasSuccessful() else "all passed"}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
