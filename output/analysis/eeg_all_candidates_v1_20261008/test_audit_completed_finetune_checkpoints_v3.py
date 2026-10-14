"""Stdlib synthetic tests; no real tensor/checkpoint/model/evaluation loads."""
from copy import deepcopy
import ast
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
import types
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name("audit_completed_finetune_checkpoints_v3.py")
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
        self.contract = {"trainable_parameter_names": ["encoder.layer.weight", "head.weight", "head.bias"]}

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
        if name != "best.pt":
            data["state_dict"] = {name: types.SimpleNamespace(shape=shape) for name, shape in
                                  (("head.weight", (2, 3)), ("head.bias", (2,)), ("encoder.layer.weight", (3, 3)))}
            data["optimizer_state_dict"] = {"param_groups": [{"group_name": "head", "params": [0, 1]}, {"group_name": "backbone", "params": [2]}], "state": {}}
            if name == "latest.pt":
                data["optimizer_state_dict"]["state"] = {i: {key: types.SimpleNamespace(shape=data["state_dict"][name].shape)
                     for key in ("exp_avg", "exp_avg_sq")} for i, name in enumerate(("head.weight", "head.bias", "encoder.layer.weight"))}
        return data

    def test_actual_optimizer_group_names_ids_and_moment_shapes(self):
        for name in ("initial.pt", "latest.pt"):
            payload = self.payload(name)
            result = A.optimizer_group_boundary(payload, self.contract)
            self.assertEqual(["head.weight", "head.bias", "encoder.layer.weight"], result["serialized_id_parameter_names"])
        for mutation in ("name", "order", "head_count", "ids", "moment"):
            payload = self.payload("latest.pt")
            groups = payload["optimizer_state_dict"]["param_groups"]
            if mutation == "name": groups[1]["group_name"] = "unrecognized"
            elif mutation == "order": groups.reverse()
            elif mutation == "head_count": groups[0]["params"], groups[1]["params"] = [0], [1, 2]
            elif mutation == "ids": groups[1]["params"] = [1]
            else: payload["optimizer_state_dict"]["state"][2]["exp_avg"].shape = (99,)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                A.optimizer_group_boundary(payload, self.contract)

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
            result = A.cpu_audit(types.SimpleNamespace(deep_checkpoint=deep), self.fit, self.row, self.contract, self.metadata, {"synthetic": "lock"}, ops)
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

    def trace_fixture(self, *, unrelated=False, running=False):
        temporary = tempfile.TemporaryDirectory(dir=A.S.REPORT, prefix="ft_v3_read_fixture_")
        root = Path(temporary.name).resolve()
        self.assertTrue(root.is_relative_to(A.S.REPORT.resolve()))
        self.addCleanup(temporary.cleanup)
        patcher = patch.object(A.S, "OUT", root)
        patcher.start()
        self.addCleanup(patcher.stop)
        path = root / "finetune_v2_attempts" / self.fit["condition"] / "a_matching" / "attempt.json"
        path.parent.mkdir(parents=True)
        record = self.trace(state="running" if running else "complete")["record"]
        path.write_text(json.dumps(record), encoding="utf-8")
        other = None
        if unrelated:
            other = path.parent.parent / "z_active_other_fold" / "attempt.json"
            other.parent.mkdir()
            data = deepcopy(record)
            data.update(fold=2 if self.fit["fold"] != 2 else 3, status="running")
            other.write_text(json.dumps(data), encoding="utf-8")
        return path, other

    def test_unrelated_active_record_read_once_and_never_hash_reopened(self):
        target, active = self.trace_fixture(unrelated=True)
        original = Path.read_bytes
        reads = []
        def reader(path):
            reads.append(path)
            if path == active and reads.count(active) > 1:
                raise PermissionError("simulated replacing unrelated active attempt")
            return original(path)
        with patch.object(Path, "read_bytes", reader), patch.object(A.S, "desc", side_effect=AssertionError("must hash only already-read matching bytes")):
            result = A.trace_inventory(self.fit)
        self.assertEqual(1, len(result))
        self.assertEqual(2, reads.count(target))
        self.assertEqual(1, reads.count(active))
        self.assertEqual(A.hashlib.sha256(original(target)).hexdigest(), result[0]["artifact"]["sha256"])

    def test_unrelated_permission_retries_identity_read_then_skips(self):
        target, active = self.trace_fixture(unrelated=True)
        original = Path.read_bytes
        reads = []
        def reader(path):
            reads.append(path)
            if path == active and reads.count(active) == 1:
                raise PermissionError("single transient unrelated read")
            return original(path)
        with patch.object(Path, "read_bytes", reader), patch.object(A.time, "sleep") as sleep:
            result = A.trace_inventory(self.fit)
        self.assertEqual(1, len(result))
        self.assertEqual(2, reads.count(active))
        self.assertEqual([unittest.mock.call(.02)], sleep.call_args_list)

    def test_unknown_permission_exhausts_exact_bound_and_is_not_skipped(self):
        target, _ = self.trace_fixture()
        with patch.object(Path, "read_bytes", side_effect=PermissionError("unknown identity")) as read, patch.object(A.time, "sleep") as sleep:
            with self.assertRaises(PermissionError):
                A.trace_inventory(self.fit)
        self.assertEqual(4, read.call_count)
        self.assertEqual([unittest.mock.call(value) for value in (.02, .04, .08)], sleep.call_args_list)

    def test_initial_missing_and_partial_json_are_bounded_retried(self):
        target, _ = self.trace_fixture()
        raw = target.read_bytes()
        for first in (FileNotFoundError("replace"), b'{"condition":', b'\xff'):
            with self.subTest(first=first), patch.object(Path, "read_bytes", side_effect=[first, raw, raw]) as read, patch.object(A.time, "sleep") as sleep:
                result = A.trace_inventory(self.fit)
                self.assertEqual(1, len(result))
                self.assertEqual(3, read.call_count)
                self.assertEqual(1, sleep.call_count)

    def test_unknown_invalid_json_exhausts_retry_never_omitted(self):
        self.trace_fixture()
        with patch.object(Path, "read_bytes", return_value=b'{') as read, patch.object(A.time, "sleep") as sleep:
            with self.assertRaises(json.JSONDecodeError):
                A.trace_inventory(self.fit)
        self.assertEqual((4, 3), (read.call_count, sleep.call_count))

    def test_formal_unknown_identity_fails_even_if_it_might_be_unrelated(self):
        target, _ = self.trace_fixture()
        record = json.loads(target.read_text())
        del record["fold"]
        target.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "identity cannot be omitted"):
            A.trace_inventory(self.fit)

    def test_matching_initial_transient_permission_retains_strict_binding(self):
        target, _ = self.trace_fixture()
        raw = target.read_bytes()
        with patch.object(Path, "read_bytes", side_effect=[PermissionError("transient"), raw, raw]) as read, patch.object(A.time, "sleep") as sleep:
            result = A.trace_inventory(self.fit)
        self.assertEqual(3, read.call_count)
        self.assertEqual(1, sleep.call_count)
        self.assertEqual(A.hashlib.sha256(raw).hexdigest(), result[0]["artifact"]["sha256"])

    def test_matching_second_read_failure_defers_instead_of_accepting_first(self):
        target, _ = self.trace_fixture()
        raw = target.read_bytes()
        for error in (PermissionError("busy target"), FileNotFoundError("target replacement")):
            with self.subTest(error=error), patch.object(Path, "read_bytes", side_effect=[raw, error, error, error, error]) as read, patch.object(A.time, "sleep") as sleep:
                with self.assertRaisesRegex(A.TraceDeferred, "bounded retry exhausted"):
                    A.trace_inventory(self.fit)
                self.assertEqual((5, 3), (read.call_count, sleep.call_count))

    def test_matching_bytes_change_deferred_not_retried_to_complete(self):
        target, _ = self.trace_fixture()
        raw = target.read_bytes()
        changed = json.loads(raw)
        changed["optimizer_step_returned"] -= 1
        with patch.object(Path, "read_bytes", side_effect=[raw, json.dumps(changed).encode()]) as read, patch.object(A.time, "sleep") as sleep:
            with self.assertRaisesRegex(A.TraceDeferred, "changed during byte binding"):
                A.trace_inventory(self.fit)
        self.assertEqual(2, read.call_count)
        self.assertEqual(0, sleep.call_count)

    def test_matching_running_deferred_before_second_read_or_any_checkpoint(self):
        target, _ = self.trace_fixture(running=True)
        raw = target.read_bytes()
        with patch.object(Path, "read_bytes", return_value=raw) as read:
            with self.assertRaisesRegex(A.TraceDeferred, "still running"):
                A.completed_fit_audit(None, self.fit, {}, self.registration, {}, operations())
        self.assertEqual(1, read.call_count)

    def test_post_cpu_changed_inventory_is_deferred(self):
        first = self.trace()
        for replacement in ([], [self.trace("b")]):
            with self.subTest(replacement=replacement), patch.object(A, "trace_inventory", return_value=replacement):
                with self.assertRaisesRegex(A.TraceDeferred, "inventory changed during CPU audit"):
                    A.recheck_trace_inventory(self.fit, [first])

    def test_post_cpu_running_recheck_is_deferred(self):
        path = A.S.OUT / "finetune_v2_attempts" / self.fit["condition"] / "attempt.json"
        with patch.object(A, "trace_inventory", side_effect=A.TraceDeferred(path, "matching running")):
            with self.assertRaises(A.TraceDeferred):
                A.recheck_trace_inventory(self.fit, [self.trace()])

    def test_deferred_or_unknown_never_claims_all135_and_keeps_loads(self):
        def status(condition, seed, fold):
            return {"status": "training_complete", "condition": condition, "seed": seed, "fold": fold}
        path = A.S.OUT / "finetune_v2_attempts" / self.fit["condition"] / "attempt.json"
        for failure in (A.TraceDeferred(path, "post-CPU target changed"), PermissionError("unreadable unknown identity")):
            def completed(wide, fit, status, registration, locks, ops):
                if fit["fit_id"] == self.fit["fit_id"]:
                    ops["cpu_checkpoint_loads_started"] += 3
                    ops["cpu_checkpoint_loads_returned"] += 3
                    raise failure
                return {**self.row, "fit_id": fit["fit_id"], "f2_physical_durable_boundary": {"durable_backward_calls_counted_once": 810}}
            with self.subTest(failure=failure), patch.object(A.R, "fit_status", side_effect=status), patch.object(A, "load_wide", return_value=object()), \
                    patch.object(A, "completed_fit_audit", side_effect=completed):
                result = A.audit()
            self.assertEqual(135, result["training_complete_finetune_observed"])
            self.assertEqual(134, result["verified_completed_finetune_fits"])
            self.assertEqual(134 * 120, result["durable_optimizer_updates_verified_subtotal"])
            self.assertEqual(3, result["checkpoint_loads"])
            self.assertFalse(result["all_135_finetune_checkpoints_verified"])
            if isinstance(failure, A.TraceDeferred):
                self.assertEqual(3, result["deferred_completed_fits"][0]["checkpoint_loads"])
                self.assertEqual([], result["errors"])
            else:
                self.assertEqual("failed", result["status"])
                self.assertEqual(1, len(result["errors"]))

    def test_cpu_math_group_and_counter_functions_unchanged_from_v2(self):
        old = ast.parse(SOURCE.with_name("audit_completed_finetune_checkpoints_v2.py").read_text(encoding="utf-8"))
        new = ast.parse(SOURCE.read_text(encoding="utf-8"))
        old_functions = {node.name: node for node in old.body if isinstance(node, ast.FunctionDef)}
        new_functions = {node.name: node for node in new.body if isinstance(node, ast.FunctionDef)}
        for name in ("require", "integer", "seconds", "delegated_sources", "load_wide", "finetune_matrix", "trace_boundary",
                     "optimizer_group_boundary", "checkpoint_payload", "cpu_audit"):
            with self.subTest(function=name):
                self.assertEqual(ast.dump(old_functions[name]), ast.dump(new_functions[name]))

    def test_v2_sources_and_real_failed_evidence_preserved(self):
        expected = {"audit_completed_finetune_checkpoints_v2.py": "12bbb62d025bb2bbacfefa68677a9707185ae5943927b245b1e817a0124802bb",
                    "test_audit_completed_finetune_checkpoints_v2.py": "7976f77e5ebb7c55cb455cc3b8b890c21cecc841417b13b4fe3508bd71979df8",
                    "completed_finetune_checkpoint_audit_v2_1791516059601364100.json": "b460ebc0f06730b4340463855d4636f3c9b8bf5ce4e822adfc04ffb7ad21ee0c"}
        for name, sha in expected.items():
            self.assertEqual(sha, A.S.digest(SOURCE.with_name(name)))
        self.assertEqual(A.WIDE_SHA256, A.delegated_sources()["wide_acceptance"]["sha256"])


if __name__ == "__main__":
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PartialFineTuneTests))
    record = {"source": A.S.desc(SOURCE), "tests": A.S.desc(Path(__file__)), "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "passed": result.wasSuccessful(),
              "scope": "synthetic stdlib tests and registered JSON identities only; fake tensor loader does not validate actual tensors",
              "actual_gpu_calls": 0, "actual_models_constructed": 0, "actual_checkpoints_loaded": 0,
              "test_gate_called": False, "outer_probabilities_read": False, "output": stream.getvalue(),
              "regression_evidence": A.S.desc(SOURCE.with_name("completed_finetune_checkpoint_audit_v2_1791516059601364100.json")),
              "preserved_v2_source": A.S.desc(SOURCE.with_name("audit_completed_finetune_checkpoints_v2.py")),
              "preserved_v2_tests": A.S.desc(SOURCE.with_name("test_audit_completed_finetune_checkpoints_v2.py")),
              "actual_completed_fit_audit_run_by_this_test": False}
    path = A.S.REPORT / ("completed_ft_audit_v3_stdlib_tests_" + str(time.time_ns()) + ".json")
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(record, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    print(stream.getvalue())
    print(json.dumps({"audit": A.S.desc(path), "passed": result.wasSuccessful()}, ensure_ascii=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
