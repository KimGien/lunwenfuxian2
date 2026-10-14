"""Stdlib synthetic tests; no real tensor/checkpoint/model/evaluation loads."""
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name("audit_completed_finetune_checkpoints.py")
spec = importlib.util.spec_from_file_location("ft_partial_audit_tests_target", SOURCE)
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def operations():
    return {"cpu_checkpoint_loads_started": 0, "cpu_checkpoint_loads_returned": 0,
            "model_construction_attempts_blocked": 0, "gpu_initialization_attempts_blocked": 0}


class PartialFineTuneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registration = A.R.load_registration(verify_sources=False)
        cls.fits = A.finetune_matrix(cls.registration)

    def setUp(self):
        self.fit = deepcopy(self.fits[0])
        self.metadata = {"epochs": 30, "optimizer_updates": 120, "backward_calls": len(self.fit["train_ids"]) * 30,
                         "head_optimizer_updates": 120, "backbone_optimizer_updates": 80}
        self.row = {"epochs": 30, "best_epoch": 10, "updates": 120, "fit_id": self.fit["fit_id"]}

    def trace(self, suffix="a", *, updates=None, backwards=None, state="complete"):
        updates = self.metadata["optimizer_updates"] if updates is None else updates
        backwards = self.metadata["backward_calls"] if backwards is None else backwards
        record = {"condition": self.fit["condition"], "kind": "formal", "seed": self.fit["seed"], "fold": self.fit["fold"],
                  "status": state, "started_at": "2026-10-09T08:00:00+08:00", "finished_at": "2026-10-09T08:10:00+08:00", "seconds": 600.0,
                  "implementation": {"version": 2, "wrapper_source": A.S.desc(A.F2_SOURCE), "delegated_runner_source": A.S.desc(A.F1_SOURCE),
                    "luna_adapter_source": A.S.desc(A.F2_SOURCE.parent / "adapters/luna_v2.py"),
                    "training_math_sampler_warmup_optimizer_stop_and_resume_unchanged": True}}
        for kind in A.COUNTERS:
            value = updates if kind == "optimizer_step" else backwards if kind == "backward" else backwards + (10 if backwards else 0)
            record[kind + "_started"] = record[kind + "_returned"] = value
        return {"artifact": {"path": "synthetic/" + suffix + "/attempt.json", "sha256": suffix * 64}, "record": record}

    def test_exact_registered_135_not_glob_or_filename_count(self):
        self.assertEqual(135, len(self.fits))
        self.assertEqual(9, len({fit["condition"] for fit in self.fits}))
        altered = deepcopy(self.registration)
        index = next(i for i, fit in enumerate(altered["fits"]) if fit["fit_id"] == self.fit["fit_id"])
        altered["fits"][index] = deepcopy(self.fits[1])
        with self.assertRaisesRegex(ValueError, "matrix"):
            A.finetune_matrix(altered)

    def test_pending_does_not_load_wide_torch_or_nonft_status(self):
        visited = []
        def status(condition, seed, fold):
            visited.append(A.R.fit_id(condition, seed, fold))
            return {"status": "engineering_pending"}
        with patch.object(A.R, "fit_status", side_effect=status), patch.object(A, "load_wide", side_effect=AssertionError("no delegate while pending")), \
                patch.object(A, "completed_fit_audit", side_effect=AssertionError("no checkpoint audit while pending")):
            result = A.audit()
        self.assertEqual("pending", result["status"])
        self.assertEqual({fit["fit_id"] for fit in self.fits}, set(visited))
        self.assertEqual(135, len(result["pending_fits"]))
        self.assertEqual(0, result["checkpoint_loads"])
        self.assertFalse(result["all_135_finetune_checkpoints_verified"])
        self.assertFalse(result["test_gate_called"])

    def test_one_completed_is_partial_and_counts_durable_once(self):
        def status(condition, seed, fold):
            return {"status": "training_complete" if A.R.fit_id(condition, seed, fold) == self.fit["fit_id"] else "ready",
                    "condition": condition, "seed": seed, "fold": fold}
        def completed(wide, fit, status, registration, locks, ops):
            self.assertEqual(self.fit["fit_id"], fit["fit_id"])
            ops["cpu_checkpoint_loads_started"] += 3
            ops["cpu_checkpoint_loads_returned"] += 3
            return {**self.row, "f2_physical_durable_boundary": {"durable_backward_calls_counted_once": 810}}
        with patch.object(A.R, "fit_status", side_effect=status), patch.object(A, "load_wide", return_value=object()), \
                patch.object(A, "completed_fit_audit", side_effect=completed):
            result = A.audit()
        self.assertEqual("partial_verified", result["status"])
        self.assertEqual((1, 134, 3, 120), (result["logical_completed_fits_counted"], len(result["pending_fits"]), result["checkpoint_loads"], result["durable_optimizer_updates_verified_subtotal"]))
        self.assertFalse(result["all_135_finetune_checkpoints_verified"])
        self.assertFalse(result["full_360_plan_completion_claimed"])

    def test_cpu_failure_operations_still_reported(self):
        def status(condition, seed, fold):
            return {"status": "training_complete" if A.R.fit_id(condition, seed, fold) == self.fit["fit_id"] else "ready",
                    "condition": condition, "seed": seed, "fold": fold}
        def fail(wide, fit, status, registration, locks, ops):
            ops["cpu_checkpoint_loads_started"] = ops["cpu_checkpoint_loads_returned"] = 1
            raise ValueError("synthetic incompatible checkpoint")
        with patch.object(A.R, "fit_status", side_effect=status), patch.object(A, "load_wide", return_value=object()), \
                patch.object(A, "completed_fit_audit", side_effect=fail):
            result = A.audit()
        self.assertEqual("failed", result["status"])
        self.assertEqual(1, result["checkpoint_loads"])
        self.assertEqual(0, result["durable_optimizer_updates_verified_subtotal"])

    def test_fresh_trace_covers_durable(self):
        result = A.trace_boundary(self.fit, self.metadata, [self.trace()])
        self.assertEqual(120, result["recorded_physical_call_subtotals"]["optimizer_step_returned"])
        self.assertEqual(120, result["durable_optimizer_updates_counted_once"])
        self.assertEqual(0, result["optimizer_returns_above_durable_known_lower_bound"])
        self.assertIsNone(result["physical_optimizer_calls_exact"])

    def test_failed_then_resume_retains_extra_without_double_count(self):
        old = self.trace("a", updates=20, backwards=5 * len(self.fit["train_ids"]), state="failed")
        resumed = self.trace("b", updates=104, backwards=26 * len(self.fit["train_ids"]))
        result = A.trace_boundary(self.fit, self.metadata, [old, resumed])
        self.assertEqual(124, result["recorded_physical_call_subtotals"]["optimizer_step_returned"])
        self.assertEqual(120, result["durable_optimizer_updates_counted_once"])
        self.assertEqual(4, result["optimizer_returns_above_durable_known_lower_bound"])
        self.assertEqual(len(self.fit["train_ids"]), result["backward_returns_above_durable_known_lower_bound"])
        self.assertFalse(result["legacy_failure_counters_added_to_f2"])
        self.assertTrue(all(item["resume_durable_baseline"] is None for item in result["attempts"]))

    def test_zero_step_success_can_be_recheck(self):
        result = A.trace_boundary(self.fit, self.metadata, [self.trace(), self.trace("b", updates=0, backwards=0)])
        self.assertEqual(2, len(result["attempts"]))
        self.assertEqual(1, result["logical_completed_fits_counted"])
        self.assertEqual(120, result["recorded_physical_call_subtotals"]["optimizer_step_returned"])

    def test_started_not_returned_not_counted_as_done(self):
        failed = self.trace("b", updates=0, backwards=0, state="failed")
        for kind in A.COUNTERS:
            failed["record"][kind + "_started"] = 1
        result = A.trace_boundary(self.fit, self.metadata, [self.trace(), failed])
        self.assertEqual(121, result["recorded_physical_call_subtotals"]["optimizer_step_started"])
        self.assertEqual(120, result["recorded_physical_call_subtotals"]["optimizer_step_returned"])
        self.assertEqual(0, result["optimizer_returns_above_durable_known_lower_bound"])

    def test_identical_copied_trace_is_alias(self):
        first = self.trace()
        copied = deepcopy(first)
        copied["artifact"]["path"] = "synthetic/copied/attempt.json"
        result = A.trace_boundary(self.fit, self.metadata, [first, copied])
        self.assertEqual(1, len(result["attempts"]))
        self.assertEqual(1, len(result["duplicate_content_aliases"]))
        self.assertEqual(120, result["recorded_physical_call_subtotals"]["optimizer_step_returned"])
        with self.assertRaisesRegex(ValueError, "path supplied twice"):
            A.trace_boundary(self.fit, self.metadata, [first, first])

    def test_missing_or_inadequate_trace_rejected(self):
        with self.assertRaisesRegex(ValueError, "no observed"):
            A.trace_boundary(self.fit, self.metadata, [])
        with self.assertRaisesRegex(ValueError, "do not cover"):
            A.trace_boundary(self.fit, self.metadata, [self.trace(updates=100)])

    def test_trace_mutations_rejected(self):
        mutations = [lambda r: r.update(optimizer_step_returned=True), lambda r: r.update(backward_started=-1),
                     lambda r: r.update(seconds=float("nan")), lambda r: r.update(fold=True),
                     lambda r: r.update(optimizer_step_started=121), lambda r: r.update(encode_returned=0),
                     lambda r: r.update(condition="not-the-fit"), lambda r: r["implementation"].update(wrapper_source={"path": "other.py", "sha256": "0" * 64})]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                item = self.trace()
                mutate(item["record"])
                with self.assertRaises((ValueError, KeyError)):
                    A.trace_boundary(self.fit, self.metadata, [item])

    def test_running_completed_wrapper_is_deferred_before_any_checkpoint(self):
        with patch.object(A, "trace_inventory", return_value=[self.trace(state="running")]):
            result = A.completed_fit_audit(None, self.fit, {}, self.registration, {}, operations())
        self.assertTrue(result["deferred"])
        self.assertEqual(0, result["checkpoint_loads"])

    def payload(self, name):
        data = {"condition": self.fit["condition"], "seed": self.fit["seed"], "fold": self.fit["fold"],
                "registry": A.R.registration_descriptor(), "condition_source_lock": {"synthetic": "lock"}}
        if name == "initial.pt":
            data.update(epoch=0, optimizer_updates=0, backward_calls=0)
        elif name == "latest.pt":
            data.update(epoch=30, optimizer_updates=120, backward_calls=self.metadata["backward_calls"],
                        history=[{"optimizer_updates": 120, "backward_calls": self.metadata["backward_calls"]}])
        else:
            data["best_epoch"] = 10
        return data

    def test_payload_counters_identity_selected_does_not_inherit_terminal(self):
        for name in ("initial.pt", "latest.pt", "best.pt"):
            payload = self.payload(name)
            result = A.checkpoint_payload(Path(name), payload, self.fit, self.row, self.metadata, {"synthetic": "lock"}, A.R.registration_descriptor())
            if name == "best.pt":
                self.assertIsNone(result["optimizer_updates"])
            payload["fold"] = 999
            with self.assertRaisesRegex(ValueError, "identity/source"):
                A.checkpoint_payload(Path(name), payload, self.fit, self.row, self.metadata, {"synthetic": "lock"}, A.R.registration_descriptor())
        payload = self.payload("latest.pt")
        payload["optimizer_updates"] = 121
        with self.assertRaisesRegex(ValueError, "durable counters"):
            A.checkpoint_payload(Path("latest.pt"), payload, self.fit, self.row, self.metadata, {"synthetic": "lock"}, A.R.registration_descriptor())

    def fake_torch(self):
        module = types.ModuleType("torch")
        module.load = lambda path, **kwargs: self.payload(Path(path).name)
        module.cuda = types.SimpleNamespace(_lazy_init=lambda: None)
        module.nn = types.SimpleNamespace(Module=type("SyntheticModule", (), {}))
        return module

    def test_cpu_wrapper_observes_same_three_loads(self):
        torch = self.fake_torch()
        def deep(fit, row, contract, metadata):
            folder = A.R.fit_directory(fit["condition"], fit["seed"], fit["fold"])
            for name in ("initial.pt", "latest.pt", "best.pt"):
                torch.load(folder / name, map_location="cpu", weights_only=False)
            return {"fit_id": fit["fit_id"], "cpu_tensors_verified": True, "model_constructions": 0}
        ops = operations()
        with patch.dict(sys.modules, {"torch": torch}), patch.object(A.S, "desc", side_effect=lambda p: {"path": str(p), "sha256": "synthetic"}):
            result = A.cpu_audit(types.SimpleNamespace(deep_checkpoint=deep), self.fit, self.row, {}, self.metadata, {"synthetic": "lock"}, ops)
        self.assertEqual(3, result["checkpoint_loads"])
        self.assertEqual(3, ops["cpu_checkpoint_loads_returned"])
        self.assertEqual(0, ops["gpu_initialization_attempts_blocked"])

    def test_cpu_wrapper_blocks_gpu_location_models_and_cuda(self):
        for forbidden in ("load_cuda", "model", "gpu"):
            with self.subTest(forbidden=forbidden):
                torch = self.fake_torch()
                def deep(*args):
                    if forbidden == "load_cuda":
                        torch.load(A.R.fit_directory(self.fit["condition"], self.fit["seed"], self.fit["fold"]) / "latest.pt", map_location="cuda", weights_only=False)
                    elif forbidden == "model":
                        torch.nn.Module()
                    else:
                        torch.cuda._lazy_init()
                ops = operations()
                with patch.dict(sys.modules, {"torch": torch}), patch.object(A.S, "desc", side_effect=lambda p: {"path": str(p), "sha256": "synthetic"}):
                    with self.assertRaises((ValueError, RuntimeError)):
                        A.cpu_audit(types.SimpleNamespace(deep_checkpoint=deep), self.fit, self.row, {}, self.metadata, {"synthetic": "lock"}, ops)
                self.assertEqual(0, ops["cpu_checkpoint_loads_returned"])


if __name__ == "__main__":
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PartialFineTuneTests))
    record = {"source": A.S.desc(SOURCE), "tests": A.S.desc(Path(__file__)), "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "passed": result.wasSuccessful(),
              "scope": "synthetic stdlib tests and registered JSON identities only; fake tensor loader does not validate actual tensors",
              "actual_gpu_calls": 0, "actual_models_constructed": 0, "actual_checkpoints_loaded": 0,
              "test_gate_called": False, "outer_probabilities_read": False, "output": stream.getvalue()}
    path = A.S.REPORT / ("completed_ft_audit_stdlib_tests_" + str(time.time_ns()) + ".json")
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(record, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    print(stream.getvalue())
    print(json.dumps({"audit": A.S.desc(path), "passed": result.wasSuccessful()}, ensure_ascii=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
