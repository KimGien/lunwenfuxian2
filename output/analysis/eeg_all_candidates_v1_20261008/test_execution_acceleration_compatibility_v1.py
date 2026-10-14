"""Pure stdlib execution-contract/CPU-auditor/cost regression tests."""
import ast
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


A = module("compat_auditor_v4", ROOT / "audit_completed_finetune_checkpoints_v4.py")
T = module("compat_old_tests", ROOT / "test_audit_completed_finetune_checkpoints_v3.py")
C = module("compat_cost_v5", ROOT / "cost_aggregation_v5.py")
W = module("compat_wide_v4", ROOT / "audit_full_plan_acceptance_v4.py")
T.A, T.SOURCE = A, Path(A.__file__)
X = A.X
from tridep_ccml.eeg_all_candidates import finetune_v3 as F3


class CompatibilityTests(T.PartialFineTuneTests):
    def test_cpu_math_group_and_counter_functions_unchanged_from_v2(self):
        old = ast.parse((ROOT / "audit_completed_finetune_checkpoints_v3.py").read_text(encoding="utf8"))
        new = ast.parse(Path(A.__file__).read_text(encoding="utf8"))
        before = {n.name: n for n in old.body if isinstance(n, ast.FunctionDef)}
        after = {n.name: n for n in new.body if isinstance(n, ast.FunctionDef)}
        for name in ("require", "integer", "seconds", "finetune_matrix", "optimizer_group_boundary", "checkpoint_payload", "cpu_audit"):
            self.assertEqual(ast.dump(before[name]), ast.dump(after[name]), name)
        legacy = deepcopy(after["legacy_trace_boundary"])
        legacy.name = "trace_boundary"
        self.assertEqual(ast.dump(before["trace_boundary"]), ast.dump(legacy))

    def execution(self):
        return {"schema": X.SCHEMA, "version": 3, "observer_version": 3,
                "source": X.descriptor(Path(F3.__file__)), "delegated_finetune_implementation": X.f2_provenance(),
                "condition": self.fit["condition"], "seed": self.fit["seed"], "fold": self.fit["fold"],
                "registration": {"path": "synthetic/registration.json", "sha256": "a" * 64},
                "diagnostics": {"window_chunk": 8, "qualification": None, "bitwise_equivalence_required": True}}

    def buffered(self, *, status="complete", updates=None, backwards=None, reload_only=False):
        execution = self.execution()
        latest = (A.R.fit_directory(self.fit["condition"], self.fit["seed"], self.fit["fold"]) / "latest.pt").relative_to(X.S.WORK).as_posix()
        with tempfile.TemporaryDirectory(dir=X.S.WORK / "output", prefix="t_f3_") as temp, patch.object(F3.S, "OUT", Path(temp)):
            self.assertTrue(Path(temp).resolve().is_relative_to((X.S.WORK / "output").resolve()))
            trace = F3.BufferedAttempt(self.fit["condition"], self.fit["seed"], self.fit["fold"], execution)
            trace.flush("fit_before")
            updates = self.metadata["optimizer_updates"] if updates is None else updates
            backwards = self.metadata["backward_calls"] if backwards is None else backwards
            for kind in X.COUNTERS:
                n = updates if kind == "optimizer_step" else backwards if kind == "backward" else backwards + 10 if backwards else 0
                if reload_only: n = 2 if kind == "forward" else 16 if kind in ("encode", "encoder_forward") else 120 if kind == "encoder_windows" else 0
                trace.record[kind + "_started"] = trace.record[kind + "_returned"] = n
            if updates:
                trace.record["latest_durable_checkpoint"] = {"path": latest,
                    "epoch": self.metadata["epochs"], "optimizer_updates": self.metadata["optimizer_updates"], "backward_calls": self.metadata["backward_calls"],
                    "checkpoint_save_returned": True, "checkpoint_hash": None, "hash_scope": "not_rehashed_each_epoch; final_checkpoint_audit_binds_bytes"}
            if status == "running": trace.flush("subject_update_after")
            else: trace.finish(ValueError("synthetic") if status == "failed" else None)
            record = json.loads(trace.path.read_text(encoding="utf8"))
        return {"artifact": {"path": "synthetic/f3/attempt.json", "sha256": "c" * 64}, "record": record}

    def validate(self, item):
        with patch.object(X, "load_execution_acceleration", return_value=self.execution()):
            return X.verify_observer_record(item["record"], item["artifact"], self.fit)

    def test_real_buffered_state_machine_shared_complete_verifier(self):
        proof = self.validate(self.buffered())
        self.assertTrue(proof["counter_snapshot_exact_for_process"])
        self.assertFalse(proof["unflushed_tail_unknown"])

    def test_running_snapshot_does_not_mean_zero_lost_tail(self):
        item = self.buffered(status="running")
        proof = self.validate(item)
        self.assertIsNone(proof["unflushed_tail_counters"])
        self.assertTrue(proof["persisted_counters_are_lower_bounds"])
        self.assertFalse(proof["counter_snapshot_exact_for_process"])

    def test_running_tail_zero_fill_rejected(self):
        item = self.buffered(status="running")
        item["record"]["unflushed_tail_counters"] = dict.fromkeys(X.COUNTER_FIELDS, 0)
        with self.assertRaises(ValueError): self.validate(item)

    def test_bool_counter_and_fake_flush_rejected(self):
        for key, value in (("optimizer_step_started", True), ("flush_sequence", True), ("flush_counts", {"fit_returned": 3}), ("dirty_flag_scope", "all_future_work")):
            item = self.buffered()
            item["record"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate(item)

    def test_failed_unreturned_call_is_not_completed_work(self):
        item = self.buffered(status="failed")
        item["record"]["optimizer_step_started"] += 1
        item["record"]["durable_counter_snapshot"]["optimizer_step_started"] += 1
        item["record"]["started_without_observed_return"]["optimizer_step"] = 1
        proof = self.validate(item)
        self.assertTrue(proof["started_not_returned_calls_are_not_completed_work"])
        item["record"]["status"] = "complete"
        item["record"]["last_flush_reason"] = "fit_returned"
        with self.assertRaises(ValueError): self.validate(item)

    def test_actual_f3_math_provenance_tamper_rejected(self):
        item = self.buffered()
        item["record"]["implementation"]["delegated_runner_source"] = {"path": "fake.py", "sha256": "0" * 64}
        with self.assertRaises(ValueError): self.validate(item)

    def test_f3_completed_and_old_failed_counts_not_double_counted(self):
        item = self.buffered()
        metadata = {**self.metadata, "execution_acceleration": self.execution(), "versioned_finetune_implementation": X.f2_provenance(), "window_chunk": 8, "micro_subjects": 1}
        old = self.trace("a", updates=4, backwards=27, state="failed")
        with patch.object(X, "load_execution_acceleration", return_value=self.execution()):
            result = A.trace_boundary(self.fit, metadata, [old, item])
        self.assertEqual([2, 3], result["actual_observer_versions"])
        self.assertEqual(self.metadata["optimizer_updates"] + 4, result["recorded_physical_call_subtotals"]["optimizer_step_returned"])
        self.assertEqual(self.metadata["optimizer_updates"], result["durable_optimizer_updates_counted_once"])
        self.assertIsNone(result["physical_optimizer_calls_exact"])

    def test_f3_training_cannot_hide_behind_old_metadata(self):
        with patch.object(X, "load_execution_acceleration", return_value=self.execution()), self.assertRaisesRegex(ValueError, "metadata execution"):
            A.trace_boundary(self.fit, self.metadata, [self.buffered()])

    def test_f3_zero_step_existing_f2_completion_recheck(self):
        item = self.buffered(updates=0, backwards=0)
        with patch.object(X, "load_execution_acceleration", return_value=self.execution()):
            result = A.trace_boundary(self.fit, self.metadata, [self.trace(), item])
        self.assertTrue(result["actual_execution_observations"][-1]["zero_call_existing_completion_recheck"])

    def test_f3_reload_only_existing_metadata_recovery(self):
        item = self.buffered(updates=0, backwards=0, reload_only=True)
        with patch.object(X, "load_execution_acceleration", return_value=self.execution()):
            result = A.trace_boundary(self.fit, self.metadata, [self.trace(), item])
        self.assertTrue(result["actual_execution_observations"][-1]["no_new_training_completion_or_metadata_recovery"])
        self.assertFalse(result["actual_execution_observations"][-1]["zero_call_existing_completion_recheck"])

    def test_wide_original_math_and_cpu_functions_preserved(self):
        old = ast.parse((ROOT / "audit_full_plan_acceptance_v3.py").read_text(encoding="utf8"))
        new = ast.parse(Path(W.__file__).read_text(encoding="utf8"))
        before = {n.name: n for n in old.body if isinstance(n, ast.FunctionDef)}
        after = {n.name: n for n in new.body if isinstance(n, ast.FunctionDef)}
        for name in ("policy_curve", "deep_checkpoint", "verify_registered_identity", "verify_exported_tables", "cost_recovery_audit", "semantic_review_audit"):
            self.assertEqual(ast.dump(before[name]), ast.dump(after[name]), name)
        legacy = deepcopy(after["legacy_check_fit"])
        legacy.name = "check_fit"
        self.assertEqual(ast.dump(before["check_fit"]), ast.dump(legacy))

    def test_cost_lower_bound_resets_exact_category_total(self):
        item = self.buffered(status="running")
        normalized = C.load_v4().normalized(item["record"])
        row = {"artifact": {"path": "tridep_ccml/experiments/finetune_v2_attempts/a/attempt.json", "sha256": "d" * 64},
               "category": "formal_training_attempt", **normalized}
        result = {"source": X.descriptor(C.DELEGATE), "cost_records": [row], "rules": [],
                  "category_observations": {"formal_training_attempt": {"counter_totals": {"optimizer_returned": {"known_observed_subtotal": 120, "exact_category_total": 120}}}}}
        with patch.object(C, "bound_json", return_value=item["record"]), patch.object(X, "load_execution_acceleration", return_value=self.execution()):
            updated = C.augment(result)
        field = updated["category_observations"]["formal_training_attempt"]["counter_totals"]["optimizer_returned"]
        self.assertIsNone(field["exact_category_total"])
        self.assertEqual(120, field["known_observed_subtotal"])
        self.assertTrue(updated["execution_observer_verification"]["observations"][0]["unflushed_tail_unknown"])

    def baseline_registration_fixture(self):
        condition = self.fit["condition"]
        binding = {"path": (X.S.OUT / "execution_acceleration_v1" / condition / "registration.json").relative_to(X.S.WORK).as_posix(), "sha256": "e" * 64}
        sources = {key: X.descriptor(Path(X.__file__).with_name(key + ".py")) for key in
                   ("finetune_v3", "finetune_v2", "finetune", "validation_batching_v1", "execution_acceleration_v1")}
        value = {"schema": X.SCHEMA, "version": 1, "status": "qualified", "registry": X.R.registration_descriptor(),
                 "single_gpu": True, "training_math_unchanged": True, "training_micro_subjects": 1, "training_window_chunk": 8,
                 "previous_execution_registration": X.descriptor(X.S.OUT / "continuation_queue_v3/registration.json"), "sources": sources,
                 "conditions": {condition: {"condition_source_lock": X.R._condition_lock(condition), "diagnostic_window_chunk": 8, "qualification": None}}}
        return binding, value

    def load_fixture(self, binding, value, **changes):
        original_checked, original_read = X.checked, X.read_bound
        with patch.object(X, "checked", side_effect=lambda b: X.S.WORK / b["path"] if b == binding else original_checked(b)), \
             patch.object(X, "read_bound", side_effect=lambda b: deepcopy(value) if b == binding else original_read(b)):
            return X.load_execution_acceleration(binding, condition=changes.get("condition", self.fit["condition"]),
                                                 seed=changes.get("seed", self.fit["seed"]), fold=changes.get("fold", self.fit["fold"]))

    def test_full_source_registered_baseline8_load_without_tensor_import(self):
        binding, value = self.baseline_registration_fixture()
        loaded = self.load_fixture(binding, value)
        self.assertEqual(8, loaded["diagnostics"]["window_chunk"])
        self.assertEqual(X.f2_provenance(), loaded["delegated_finetune_implementation"])

    def test_sidecar_unknown_source_or_unsafe_training_chunk_rejected(self):
        for kind in ("source", "batch", "unqualified32"):
            binding, value = self.baseline_registration_fixture()
            if kind == "source": value["sources"]["finetune_v3"]["sha256"] = "0" * 64
            if kind == "batch": value["training_window_chunk"] = 32
            if kind == "unqualified32": value["conditions"][self.fit["condition"]]["diagnostic_window_chunk"] = 32
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.load_fixture(binding, value)

    def test_sidecar_bool_identity_rejected(self):
        binding, value = self.baseline_registration_fixture()
        with self.assertRaises(ValueError): self.load_fixture(binding, value, fold=True)

    def test_stopped_f2_proof_retains_unknown_elapsed_and_adds_no_extra_twice(self):
        old = self.trace("a", updates=4, backwards=27, state="running")
        proof = {"passed": True, "proof": {"path": "synthetic/safe_stop.json", "sha256": "f" * 64}, "process_exit_proven": True}
        old["proven_stopped_handoff"] = proof
        with patch.object(X, "verify_stopped_attempt", return_value=proof):
            result = A.trace_boundary(self.fit, self.metadata, [old, self.trace("b")])
        self.assertEqual("stopped_by_verified_handoff", result["attempts"][0]["status"])
        self.assertIsNone(result["attempts"][0]["seconds_this_attempt"])
        self.assertEqual(self.metadata["optimizer_updates"] + 4, result["recorded_physical_call_subtotals"]["optimizer_step_returned"])
        self.assertIsNone(result["all_historical_wall_seconds_exact"])


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CompatibilityTests))
    raise SystemExit(0 if result.wasSuccessful() else 1)
