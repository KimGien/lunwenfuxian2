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
SPEC = importlib.util.spec_from_file_location("_final_cost_gate_test", REPORT / "finalize_full_plan_cost.py")
F = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(F)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class GateTests(unittest.TestCase):
    def setUp(self):
        self.fixture = Path(tempfile.mkdtemp(prefix="cost_finalizer_fixture_", dir=REPORT)).resolve()
        self.assertEqual(self.fixture.parent, REPORT.resolve())
        self.saved = F.OUT, F.QUEUE
        F.OUT = self.fixture / "out"
        F.QUEUE = F.OUT / "continuation_queue_v3"
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
        self.completion = {"passed": True, "formal_fits_verified": 360, "supervised_conditions": 23,
                           "supervised_fits_evaluated": 345, "evaluated_checkpoints": 435, "candidate_papers_count": 9}

    def tearDown(self):
        F.OUT, F.QUEUE = self.saved
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

    def test_supplementary_head_resource_requires_real_new_scope_and_zero_updates(self):
        registration = {"fits": self.fits, "configuration": {"conditions": [{"condition_id": "Demo0", "policy_key": "frozen"}]},
                        "folds": {"fold_1": {"train": ["train_person"]}}}
        write(F.OUT / "registration.json", registration)
        lock_path = F.OUT / "engineering/Demo0.json"
        write(lock_path, {"contract": {"engineering_gradient_passes": 1}})
        source = self.fixture / "synthetic_measurement_source.py"
        source.write_text("# synthetic source identity only; never executed\n", encoding="utf-8")
        value = {"passed": True, "subtype": "supplemental_head_resource_measurement", "condition": "Demo0",
                 "new_measurement_not_historical_backfill": True, "registration": F.desc(F.OUT / "registration.json"),
                 "condition_source_lock": F.desc(lock_path), "source": F.desc(source), "seconds": 0.5,
                 "peak_allocated_bytes": 1024, "optimizer_updates": 0, "input_role": "train",
                 "gradient_passes": 1, "forward_backward_seconds": 0.25,
                 "seed": 11, "fold": 1, "probe_subject_ids": ["train_person"]}
        path = F.OUT / "supervised_engineering/Demo0/measurement_synthetic.json"
        write(path, value)
        result = F.head_resource_gate(registration)
        self.assertTrue(result["passed"])
        self.assertTrue(result["verified_measurements"][0]["old_missing_seconds_remains_unknown"])
        self.assertTrue(result["verified_measurements"][0]["old_missing_peak_remains_unknown"])
        for change in ({"optimizer_updates": 1}, {"probe_subject_ids": ["test_person"]},
                       {"new_measurement_not_historical_backfill": False}, {"seconds": None},
                       {"peak_allocated_bytes": True}, {"input_role": "test"}, {"gradient_passes": 0},
                       {"forward_backward_seconds": None}):
            write(path, {**value, **change})
            self.assertFalse(F.head_resource_gate(registration)["passed"])


if __name__ == "__main__":
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(GateTests)
    names = [test.id() for test in suite]
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    value = {"passed": result.wasSuccessful(), "tests_run": result.testsRun, "failures": len(result.failures),
             "errors": len(result.errors), "seconds": time.perf_counter()-started, "tests": names,
             "test_output": stream.getvalue(), "pure_standard_library": True,
             "real_completed_report_verifier_calls": 0, "gpu_calls": 0, "model_constructions": 0,
             "raw_signal_reads": 0, "test_probability_reads": 0, "optimizer_updates": 0,
             "final_cost_artifacts_created": 0,
             "sources": [F.desc(Path(__file__)), F.desc(Path(F.__file__))]}
    target = REPORT / ("final_cost_gate_stdlib_tests_" + str(time.time_ns()) + ".json")
    with target.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"passed": value["passed"], "tests_run": result.testsRun, "evidence": F.desc(target),
                      "output": stream.getvalue() if not result.wasSuccessful() else "all passed"}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
