"""Pure-stdlib safety/acceptance tests: no models, tensors, raw signals or GPU."""
import copy
from contextlib import ExitStack, contextmanager
import importlib.util
import json
from pathlib import Path
import unittest
import uuid
from unittest.mock import patch

SOURCE = Path(__file__).with_name("audit_full_plan_acceptance_v2.py")
spec = importlib.util.spec_from_file_location("acceptance_v2_under_review", SOURCE)
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
REGISTRATION = A.S.read(A.R.REGISTRATION)


@contextmanager
def fixture_directory():
    # Retain tiny, explicitly synthetic fixtures under the authorized analysis
    # directory. Windows sandbox temporary-directory ACLs are not reliable.
    path = Path(__file__).parent.resolve() / "acceptance_test_fixtures" / uuid.uuid4().hex
    assert path.resolve().is_relative_to(Path(__file__).parent.resolve())
    path.mkdir(parents=True)
    yield str(path)


class AcceptanceTests(unittest.TestCase):
    def simulated_run(self, complete_count, *, semantic=True, mamba=True, eval_failure=False):
        # Patch only interfaces. Any accidental checkpoint/evaluator access in
        # the pending branch must trigger the mock assertions below.
        with fixture_directory() as name, ExitStack() as stack:
            root = Path(name)
            out, report = root / "out", root / "report"
            report.mkdir(); (out / "evaluation").mkdir(parents=True)
            (out / "evaluation/complete.json").write_text("{}")
            (report / "eegmamba_official_gpu_check.json").write_text(json.dumps({"passed": mamba}))
            descriptor = {"path": "fake/artifact", "sha256": "0" * 64}
            prep = {key: descriptor for key in ("plan", "protocol_source", "immutable_registration", "versioned_execution_registration")}
            prep["requirements"] = [{"id": f"R{i:02}", "requirement": str(i), "authoritative_evidence_locations": []} for i in range(1, 31)]
            real_read = A.S.read

            def read(path):
                path = Path(path)
                if path == A.PREPARATION: return prep
                if path.name == "registration.json": return {"sources": []}
                if path.name == "eegmamba_official_gpu_check.json": return real_read(path)
                return {"contract": {}}

            indexes = {(f["condition"], f["seed"], f["fold"]): i for i, f in enumerate(REGISTRATION["fits"])}
            def status(condition, seed, fold):
                i = indexes[(condition, seed, fold)]
                return {"status": "training_complete" if i < complete_count else "engineering_pending"}

            def fitrow(fit, *args):
                return {"fit_id": fit["fit_id"], "condition": fit["condition"],
                        "policy": A.P.CONDITION_BY_ID[fit["condition"]].policy_key,
                        "metadata": descriptor, "completion": descriptor, "updates": 10}

            for obj, key, value in [(A.S, "OUT", out), (A.S, "REPORT", report), (A.S, "read", read),
                                    (A.S, "checked", lambda value: root / "fake.json"), (A.S, "desc", lambda value: descriptor),
                                    (A.R, "load_registration", lambda: copy.deepcopy(REGISTRATION)),
                                    (A.R, "fit_status", status), (A.R, "_condition_lock", lambda value: descriptor),
                                    (A.R, "registration_descriptor", lambda: descriptor),
                                    (A, "verify_registered_identity", lambda value: None), (A, "check_fit", fitrow)]:
                stack.enter_context(patch.object(obj, key, value))
            deep = stack.enter_context(patch.object(A, "deep_checkpoint", return_value={"cpu_tensors_verified": True}))
            evaluation = stack.enter_context(patch.object(A, "evaluation_audit", return_value={"passed": True, "complete": descriptor},
                                                           side_effect=ValueError("simulated after-read failure") if eval_failure else None))
            stack.enter_context(patch.object(A, "final_cost_audit", return_value={"passed": True, "snapshot": descriptor}))
            if semantic:
                stack.enter_context(patch.object(A, "semantic_review_audit", return_value={"passed": True}))
            result = A.audit()
            return result, deep.call_count, evaluation.call_count

    def test_pending_full360_identity_never_enters_tensor_or_test_path(self):
        value, tensors, test_calls = self.simulated_run(203)
        self.assertEqual(value["formal_identity_count"], 360)
        self.assertEqual(value["completed_training_json_verified"], 203)
        self.assertEqual(len(value["incomplete_fits"]), 157)
        self.assertEqual((tensors, test_calls), (0, 0))
        self.assertFalse(value["outer_test_artifacts_read"])
        self.assertFalse(value["full_plan_completed"])

    def test_full_artifacts_do_not_replace_semantic_review(self):
        value, tensors, test_calls = self.simulated_run(360, semantic=False)
        self.assertEqual((tensors, test_calls), (360, 1))
        self.assertFalse(value["full_plan_completed"])
        self.assertTrue(all(r["status"] == "pending" for r in value["requirements"] if r["id"] in A.SEMANTIC_REVIEW_IDS))

    def test_one_requirement_pending_prevents_global_completion(self):
        value, _, _ = self.simulated_run(360, mamba=False)
        self.assertFalse(value["full_plan_completed"])
        self.assertEqual(next(r for r in value["requirements"] if r["id"] == "R18")["status"], "pending")

    def test_failed_evaluation_does_not_claim_no_prediction_access(self):
        value, _, calls = self.simulated_run(360, eval_failure=True)
        self.assertEqual(calls, 1)
        self.assertTrue(value["outer_test_artifacts_read"])
        self.assertFalse(value["full_plan_completed"])

    def test_complete_is_possible_only_with_all30_requirements(self):
        value, _, _ = self.simulated_run(360)
        self.assertTrue(value["full_plan_completed"])
        self.assertTrue(all(r["status"] == "verified_complete" for r in value["requirements"]))

    def test_csv_same_shape_tampering_is_rejected(self):
        rows = [{"id": "a", "probability": .25}, {"id": "b", "probability": .75}]
        with fixture_directory() as name:
            path = Path(name) / "table.csv"
            path.write_text("id,probability\na,0.25\nb,0.75\n", encoding="utf-8-sig")
            A.verify_csv(path, rows)
            path.write_text("id,probability\na,0.75\nb,0.25\n", encoding="utf-8-sig")
            with self.assertRaisesRegex(ValueError, "cells/order"):
                A.verify_csv(path, rows)

    def test_static_fit_identity_cannot_hide_reordered_role(self):
        value = copy.deepcopy(REGISTRATION)
        A.verify_registered_identity(value)
        value["fits"][0]["train_ids"] = list(reversed(value["fits"][0]["train_ids"]))
        with self.assertRaisesRegex(ValueError, "identity/role"):
            A.verify_registered_identity(value)

    def test_semantic_review_rejects_partial_or_stale_binding(self):
        with fixture_directory() as name:
            path = Path(name) / "review.json"
            path.write_text(json.dumps({"kind": "independent_final_semantic_review", "passed": True,
                                        "reviewer": "test", "binding": {}, "items": []}))
            with patch.object(A, "semantic_review_binding", return_value={"current": "binding"}):
                with self.assertRaisesRegex(ValueError, "partial, stale, or unbound"):
                    A.semantic_review_audit(path, [], {}, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
